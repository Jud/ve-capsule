//! Small scoped-thread helpers for the embarrassingly parallel screening
//! scans. Every task is pure CPU work over shared read-only state, so
//! threading changes throughput only — never results or their order.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

pub fn worker_count(len: usize) -> usize {
    if let Ok(value) = std::env::var("VE_CAPSULE_RELATION_THREADS")
        && let Ok(requested) = value.parse::<NonZeroUsize>()
    {
        return requested.get().min(len.max(1));
    }
    thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
        .min(len.max(1))
}

/// Map `task` over `0..len` on scoped worker threads, returning the results in
/// index order. A panicking task propagates out of the join, exactly as it
/// would inline.
pub fn parallel_map_indexed<T, F>(len: usize, task: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let workers = worker_count(len);
    if workers <= 1 {
        return (0..len).map(task).collect();
    }
    let chunk_size = len.div_ceil(workers);
    thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let start = worker * chunk_size;
                let end = (start + chunk_size).min(len);
                let task = &task;
                scope.spawn(move || (start..end).map(task).collect::<Vec<T>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| match handle.join() {
                Ok(results) => results,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// Screen `items` across worker threads, returning `true` as soon as any
/// contiguous chunk's `predicate` does. The common no-hit case scans every
/// chunk; a hit lets the threads that have not started yet skip their work.
/// `predicate` runs on a sub-slice and must itself be order-independent, since
/// it only needs to answer "does any element here match".
pub fn parallel_any<T, F>(items: &[T], predicate: F) -> bool
where
    T: Sync,
    F: Fn(&[T]) -> bool + Sync,
{
    let len = items.len();
    if len == 0 {
        return false;
    }
    let workers = worker_count(len);
    if workers <= 1 {
        return predicate(items);
    }
    let chunk_size = len.div_ceil(workers);
    let found = AtomicBool::new(false);
    thread::scope(|scope| {
        for chunk in items.chunks(chunk_size) {
            let found = &found;
            let predicate = &predicate;
            scope.spawn(move || {
                if found.load(Ordering::Relaxed) {
                    return;
                }
                if predicate(chunk) {
                    found.store(true, Ordering::Relaxed);
                }
            });
        }
    });
    found.into_inner()
}
