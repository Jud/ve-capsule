# ec-segve-v1 — Soundness & Fiat–Shamir Absorption Spec

**Status:** normative soundness contract for `crates/ve-capsule` (ec-segve on
secp256k1). Sole authority for the Fiat–Shamir transcript, challenge schedule,
and sub-protocol equations. Other restatements (README, design notes, comments)
are informative and **defer here on discrepancy**.

**Date:** 2026-06-11 (composition pass 2026-07-09 — register and structure only;
normative pins unchanged)

**Grounded in:** `crates/ve-capsule/src/{assembly,range_circuit,norm_arg,carry,linking,composite,opening,dleq,generators,codec,transcript}.rs`;
Cypher Stack BP++ review; Camenisch–Shoup VE template.

> [!WARNING]
> ve-capsule is a research toy (experimental, unaudited). This is the
> cryptographic contract the implementation targets — not a shipping guarantee.

---

# Overview

## What it is

**ec-segve** (elliptic-curve *segmented* verifiable encryption) seals a secret
secp256k1 scalar `m` to a recovery public key and attaches a non-interactive
proof that the ciphertext decrypts to that scalar.

In deployment, `m` is usually a FROST / threshold **share**, or one additive
piece of a share. Anyone holding the public package can **verify**. Only the
holder of the recovery secret can **open**. What the package may reveal about
`m` is limited carefully: nothing beyond the already-public commitment
`C = m·G` (formal secrecy claim in §4.4).

### Why bother

Without a proof, a sealer can encrypt garbage. Peers store the bytes in good
faith; the failure shows up only at recovery — when it is too late to fix, and
when a ransom or denial attack has maximum leverage.

Verifiable encryption moves the check to **seal time**. Packages that would not
decrypt to the claimed share fail before anyone relies on them. When several
parties each seal a piece (a **Case**), peers can audit each other from public
data alone, with the recipient offline.

### What “right share” means

The claim is not informal trust. The package’s commitment `C = m·G` must equal
the **certified** verifying share `T` from the threshold ceremony. That
equality is an explicit verify-time check in the framework (§5) — kept *outside*
the offline proof statement so the proof is about `C` while the application
still pins its certified target.

So the proof obligation is:

> The ciphertext decrypts to the discrete logarithm of `C`.

That is four claims at once (all zero-knowledge):

| # | Claim | Failure mode if omitted |
|---|---|---|
| 1 | Ciphertexts are well-formed | Verifies but never opens (ransom) |
| 2 | Every limb is small enough to recover | Verifies but BSGS cannot finish |
| 3 | Limbs reassemble to `m` as **integers** | Opens to something only congruent mod `n` |
| 4 | That integer is the dlog of `C` | Opens to the wrong share |

This is Camenisch–Shoup verifiable encryption [CS03] on secp256k1, instantiated
in the segmented-ElGamal line of [JUG]/[Groth21] (encrypt limbs, range-prove
them, bind to the dlog of a public point). The proof is Fiat–Shamir; the
challenge schedule and absorption order are fixed in §1–§3.

---

## Roadmap

Seal path (this document’s main subject):

```
m  ──segment──►  limbs v_k ──ElGamal──► (E_k, D_k) to recovery key pk
                 │
                 ├── Pedersen Com_k / Com̄_k / ComC_k
                 │
                 ├── range     (§4.1 BP++)     limbs + carries in range
                 ├── integer   (§4.2 carry)    m + m̄ = n−1 over ℤ
                 └── bind      (§4.3 linking)  same v_k in ciphertext, Com, C
                          │
                          ▼
                 one FS transcript: BP++ challenges first, then shared sigma.x
```

Opening path (separate, short): authorizer **Partial**s carry a **DLEQ** (§6)
so a bad contribution fails the proof instead of silently poisoning recovery.
That transcript is **not** the seal multi-squeeze.

---

## Construction at a glance

1. Split `m ∈ [0, n)` into `L` limbs of `ℓ` bits each (frozen values in §0).
2. Encrypt each limb under lifted ElGamal; commit limbs, complements, and
   carries under Pedersen.
3. Prove ranges with one aggregated Bulletproofs++ argument (§4.1).
4. Prove exact integer identity `m + m̄ = n − 1` with a carry chain (§4.2).
5. Link ciphertexts, commitments, and `C` with a shared-response sigma (§4.3).
6. Derive all challenges from one absorb-everything transcript (§1–§3).
7. Compose knowledge-soundness and zero-knowledge / secrecy; state non-claims
   (§4.4).

Wire size of one full capsule is fixed (~5.4 KB at frozen params); most of that
is the linking sigma and range material. After verify, recovery can drop the
proof and keep a small core or 65-byte hints — that product layer is out of
scope here (see the crate README and recovery design docs).

> **Note.** The stripped-core attestation threat model — a fabricated core
> offered to an authorizer is a static-DH oracle on its key, closed by a quorum
> signature over the canonical attestation statement — is argued in the crate
> README ("Why stripped cores need a signature"). Non-normative here.

---

## Preliminaries

Primitives as used below. Citations are in References. Normative pins start at
§0.

### Group

Arithmetic is on **secp256k1**.

- **Point** — curve-group element. Fixed base `G`, identity `O`.
- **Scalar** — integer mod group order `n` (prime, ≈ 2^256).
- **Hard dlog** — from random `v·G` you cannot recover large `v`. We rely on
  discrete log (binding / soundness), DDH (ElGamal secrecy), and SHA-256 as a
  random oracle (Fiat–Shamir).
- **Easy small dlog** — if `v ∈ [0, 2^24)`, baby-step/giant-step recovers `v`
  from `v·G`. The scheme needs both sides: large masks stay hidden; small
  limbs are intentionally recoverable after decryption.

One subtlety used later: equality **mod `n`** is not the same as equality of
integers. Curve equations only see the former; §4.2 forces the latter for `m`.

### Lifted ElGamal and segmentation

To encrypt a small value `v` to recovery key `pk = sk·G`, sample fresh `r` and
publish

```
E = r·G
D = v·G + r·pk
```

The keyholder computes `D − sk·E = v·G` and recovers `v` by BSGS. Without `sk`
(or `r`), the pair hides `v` (IND-CPA under DDH).

A full 256-bit `m` is too large for BSGS, so split into limbs:

```
m = Σ_k v_k · 2^{ℓ k} ,   k ∈ [0, L) ,   each v_k < 2^ℓ
```

(frozen `ℓ = 24`, `L = 11`). Each limb is encrypted separately. The proof must
then show every limb is small and that the limbs really form the claimed `m`.

### Pedersen commitments

```
Com = v·G + s·H
```

with fresh blinding `s` and a second generator `H`.

- **Hiding** — random `s·H` masks `v`.
- **Binding** — two openings to different `v` would yield the dlog of `H`.

So `H` is a **NUMS** point (hash-to-curve from a fixed public tag; never taken
from the wire). Range and carry proofs talk about committed values; the linking
sigma connects those commitments to the ElGamal limbs. Exact derivation and KAT
are in §0.

### Σ-protocols

Three-move proofs of knowledge: announcement → challenge → response. Schnorr
for “I know `w` with `Y = w·G`”: announce `A = a·G`, respond `z = a + x·w`,
check `z·G = A + x·Y`.

- **Special soundness** — two accepting challenges for one announcement extract
  `w`. Cheating probability per attempt ≤ `1/n`.
- **HVZK** — with `x` known in advance, transcripts are simulatable without `w`.

Maurer’s generalization [Mau09]: the same shape works for any linear map
`φ` (Pedersen openings, ElGamal payloads, multi-leg linking). Exact maps for
carry and linking are in §4.2–§4.3.

### Fiat–Shamir

The interactive challenge is replaced by a hash of protocol id, parameters,
context, statement, and announcements, reduced mod `n`. Security is argued in
the random-oracle model.

Two rules, expanded normatively in §1–§3:

1. **Absorb everything** the checks depend on (omission = Frozen Heart class).
2. **Absorb injectively** — one canonical encoding per object.

**Strong FS** includes the full statement and context, not only announcements —
that is what stops a proof being moved to another statement or session.

### Bulletproofs++ (sketch)

Limb ranges and carry booleanity are one **aggregated** reciprocal range proof
[BP++], not one sub-proof per digit:

- digits in base `d = 16` and a reciprocal identity at random `α`;
- circuit constraints folded under further challenges (including `δ`, required
  after the 2022 draft hole);
- logarithmic compression via the weighted norm linear argument.

Multi-round challenges must be ordered and absorbed correctly (§2). Soundness
here follows BP++ **as corrected by Cypher Stack** [CS-BPPP]. Full statement
shape and verifier obligations are §4.1.

---

# Contract

§0–§7 below are normative. **MUST** / **MUST NOT**, the absorption checklist,
and the equations are load-bearing. Implementation targets this fixed contract;
as-built traceability checks code against it.

| § | Pins |
|---|---|
| **0** | Notation, frozen `(ℓ,L,d,D)`, NUMS generators |
| **1** | Transcript primitive (framing, canonical points/scalars) |
| **2** | Multi-squeeze ratchet (labels, chaining, full reduce mod `n`) |
| **3** | Absorption checklist |
| **4** | Sub-protocols: BP++ range, carry chain, linking sigma, composition |
| **5** | Verifier-gate → enforcing component (traceability) |
| **6** | Contribute-DLEQ transcript (opening path) |
| **7** | Out of scope; residual normative codec rules |

## 0. Notation, parameters, and generators

> **Normative.** Notation and frozen constants. Impl: `params`, `generators`, `limbs`.

- `n` — the secp256k1 group order (prime, ≈ 2^256). `G` — the base point.
  `O` — the identity point. `Z_n` — the scalars (integers mod `n`). `←$` —
  a fresh uniform draw from `Z_n`.
- `m ∈ [0, n)` — the canonical scalar being encrypted. `C = m·G` — its
  fragment commitment, a pure-`G` point with **no `H` term** (the purity is
  what lets `C` be compared directly against a verifying share). `T` — the
  certified verifying share: the framework-held public point that `C` must
  equal at verify time (§5). `T` is deliberately not part of the proof
  statement — see §3.
- `m̄ = (n − 1) − m` — the complement, computed over the integers
  (well-defined because `m ≤ n − 1`), with limbs `v̄_k`. Proving
  `m + m̄ = n − 1` with both sides decomposed into nonnegative limbs is how
  §4.2 pins `m < n`.
- `pk = sk·G` — the recovery public key; `sk` is held by the party entitled
  to decrypt.
- `ℓ, L, d, D` — limb bit-width, limb count, BP++ digit base, digits per
  limb. Frozen consts: `ℓ=24, L=11, d=16, D=6`. Invariants: `d^D = 2^ℓ`
  (compile-time `const` assert; `D` base-`d` digits cover a limb exactly —
  `16^6 = 2^24`, zero over-cover) and `L·ℓ ≥ 256` (the limbs cover any
  256-bit scalar; `11·24 = 264`). The carry bits form a second, base-2
  digit group of exactly `L − 1 = 10` one-bit digits (§4.1). The two
  groups share the BP++ digit machinery but have separate multiplicity
  vectors (16 symbols and 2 symbols respectively).
- Per limb `k ∈ [0, L)`:
  - `v_k ∈ [0, 2^ℓ)` — the limb value; `v̄_k` — the matching limb of `m̄`.
  - `(E_k = r_k·G, D_k = v_k·G + r_k·pk)` — the ElGamal pair (mask handle,
    masked value), fresh `r_k` per limb.
  - `Com_k = v_k·G + s_k·H` and `Com̄_k = v̄_k·G + s̄_k·H` — hiding Pedersen
    commitments to the limb and its complement.
  - `d_{k,j}` — the base-16 digits of `v_k`, `j ∈ [0, D)`. Digits are
    **witness-only** in the BP++ construction: there are no per-digit
    commitments; the digit vectors live inside the proof's vector
    commitments (§4.1).
  - `c_k ∈ {0,1}` — the carry bits of the limb-wise addition `m + m̄`, with
    boundary constants `c_{-1} = 0` and `c_{L-1} = 0` (§4.2); committed as
    `ComC_k = c_k·G + g_k·H`.
- `x` — the sigma layer's shared FS challenge: the **final squeeze** of the
  multi-squeeze schedule (§2). The BP++ rounds have their own challenges
  (`α; ρ, λ, β, δ; τ; γ_1…γ_F`), squeezed earlier on the same transcript.
- `H` — the second ("hiding") generator: a NUMS point whose dlog w.r.t. `G`
  is unknown, the basis of Pedersen binding. `H` is not a parameter and is
  never read from the wire. Every party recomputes it locally by RFC 9380
  hash-to-curve over secp256k1 (suite `secp256k1_XMD:SHA-256_SSWU_RO_`)
  from the fixed domain-separation tag
  `H_DST = "ec-segve-secp256k1-v1:NUMS:H:RFC9380:secp256k1_XMD:SHA-256_SSWU_RO_"`
  and message `H_MSG = "hiding-generator"`, and pins the result by a
  known-answer test:
  `H = 02460a164ac67bea239d4995793e179a3f4adfc260e0a2074c93e83228af8a5482`
  (compressed SEC1; see `src/generators.rs`, audit SA-2026-307/322). A
  wire-supplied or known-dlog `H` would break Pedersen binding and with it
  the range and carry soundness — so `H` is a compile-time constant, not an
  input. Its absorption into the transcript (§3 item 13) is belt and
  suspenders, not the guarantee.
- **Vector generators (BP++).** The aggregated range proof needs two NUMS
  generator *vectors* beyond `(G, H)`: `g⃗` (the n-side / norm bases) and
  `h⃗` (the l-side / linear bases, with `h⃗[0] := H` — the existing NUMS
  point, unchanged — so the capsule's Pedersen commitments are themselves
  valid BP++ value commitments). Every party derives every element locally
  by the same RFC 9380 suite as `H`, with per-role, per-index
  domain-separation:
  `GVEC_DST = "ec-segve-secp256k1-v1:NUMS:GVEC:RFC9380:secp256k1_XMD:SHA-256_SSWU_RO_"`,
  message `"vector-generator-g-" ‖ BE16(i)`; likewise `HVEC_DST` /
  `"vector-generator-h-" ‖ BE16(i)` for `h⃗[1…]`. The element **counts are
  frozen constants** derived from the circuit shape (§4.1's frozen-shape
  table): never negotiated, never read from the wire, never variable. The
  whole set is pinned by a `generators_digest` KAT — the SHA-256 of the
  domain-separated, length-prefixed concatenation of every element's
  canonical SEC1 bytes — and that digest is absorbed into the transcript
  (§3 item 13a). Binding of the vector commitments rests on no party
  knowing any discrete-log relation among `G`, `H`, and these points —
  exactly the `H` argument, vectorized.

## 1. Transcript primitive

> **Normative.** Transcript injectivity (framing + canonical encodings). Impl: `transcript`, `codec`.

The **transcript** is the byte string that both prover and verifier feed to
SHA-256 to derive the challenge. It is never sent — each side reconstructs
it independently from the protocol constants, the statement, and the
proof's announcements. The rules in this section make the transcript
**injective**: one statement produces one byte string, and one byte string
parses as one statement. Every rule exists to deny two encodings of the
same object (two encodings means two challenges for one claim —
independent attempts against the `1/n` soundness bound) and to deny two
readings of the same bytes (one proof accepted for two claims).

- **Domain.** The EC transcript has its own domain string,
  `b"ve-capsule.ec-segve.secp256k1.v1"`, distinct from the CL-HSMq/BCL24 domain
  used by the sibling class-group backend. Cross-protocol challenge reuse
  is prevented by this domain plus the statement-version byte (§3 item 2),
  not by the caller `Context`.
- **Framing.** Every absorbed field is written length-prefixed — a 4-byte
  big-endian `u32` length, then the field bytes — into a running SHA-256
  state. This mirrors CL's `write_length_prefixed`; the *mechanic* is
  reused, not `bcl24::compute_challenge` (see §2 for why that function must
  not be reused). A field that is itself a list (e.g. the `L` limb
  ciphertexts) is absorbed as a 4-byte BE count followed by each element,
  length-prefixed, in index order.
- **Points** are absorbed as their re-serialized canonical compressed SEC1
  bytes — exactly 33 bytes for every point. The identity encoding is
  byte-exact: 33 zero bytes (`[0u8; 33]`), not SEC1's 1-byte `0x00` and not
  a 65-byte form. The strict parser (`src/codec.rs`) accepts the all-zero
  33-byte buffer as the identity and rejects every other identity
  representation (the 1-byte `0x00`, `[0u8; 65]`, any non-33-byte length)
  with `Error::PointDecode` before any decode or absorb; non-identity
  points are the standard `0x02`/`0x03` compressed form with on-curve and
  `x < p` enforced. This fixed-length, one-encoding-per-point rule is what
  makes absorption injective. Never absorb the received wire bytes:
  re-serialize from the parsed point, so that two distinct byte strings for
  the same group element cannot produce two challenges. (Do not route
  through `generic_ec::Point::from_bytes`, whose 0.4.5 decoder collapses
  any all-zero buffer of any length to the identity — SA-2026-333.)
- **Scalars** are absorbed as 32-byte big-endian, and must be canonical:
  every scalar decoded from the wire (sigma responses `z_•`, the BP++
  clear residual vectors `l⃗`/`n⃗`) MUST be range-checked `< n` and
  rejected otherwise (`Error::PointDecode`, the strict-decode error —
  `src/assembly.rs`) before any use or absorption. This forecloses the
  `z + n` malleability: a 32-byte value `≥ n` that reduces to the same
  residue would pass the algebra but encode (and hash) differently —
  two wire encodings for one accepting proof.

## 2. Challenge derivation — the labeled multi-squeeze ratchet

> **Normative.** Multi-squeeze ratchet: absorb-before-squeeze, chaining, final `sigma.x`. Impl: `transcript`.

BP++ is multi-round, so the transcript squeezes **several** challenges,
governed by a **ratchet** with three invariants:

1. **Absorb-before-squeeze, per challenge.** Every public input and every
   prover message of the current and all earlier segments is absorbed
   before each squeeze. A field absorbed late, or not at all, is a field
   the prover can pick after seeing that challenge (Frozen-Heart). §3
   freezes exactly what is absorbed before each squeeze.
2. **Chaining.** Every challenge depends on all earlier absorptions,
   labels, *and* earlier challenges — a prover cannot grind one round in
   isolation.
3. **Final squeeze.** The sigma challenge `x` (label `sigma.x`) is the
   LAST squeeze, and nothing prover-chosen is absorbed after it. The
   implementation enforces
   this at the type level: the terminal squeeze consumes the transcript
   (`finalize(self)`), exactly as `challenge(self)` did before.

**Byte-pinned ratchet.** Three domains, all distinct from every other
domain string in the crate:

```
T_DOMAIN = b"ve-capsule.ec-segve.secp256k1.v1"             (seed, §1 — unchanged)
R_DOMAIN = b"ve-capsule.ec-segve.secp256k1.v1.ratchet"
C_DOMAIN = b"ve-capsule.ec-segve.secp256k1.v1.challenge"
```

State: one running SHA-256 hasher, seeded by absorbing `T_DOMAIN` framed
(`LP(x)` = 4-byte BE length ‖ bytes, exactly §1's framing); `absorb_*`
append frames as before. Then `squeeze(label)`:

```
d := SHA256-finalize(running state)                          (32 bytes)
c := SHA256( LP(C_DOMAIN) ‖ LP(label) ‖ LP(d) )  reduced mod n
running state := fresh SHA-256 absorbing
                 LP(R_DOMAIN) ‖ LP(label) ‖ LP(d) ‖ LP(c as 32-byte BE)
```

- The challenge/ratchet domain split means a challenge value can never
  alias a chaining value; the length-prefixed label means no two labels
  collide as byte strings.
- **Labels, frozen:** `b"alpha"`, `b"rho"`, `b"lambda"`, `b"beta"`,
  `b"delta"`, `b"tau"`, `b"gamma" ‖ BE16(i)` (one per WNLA fold round,
  `i` the verifier's own loop counter, never prover-supplied), `b"sigma.x"`.
  `ρ, λ, β, δ` are **four sequential squeezes in that pinned order** —
  each ratchets the state — never four reads of one digest. (`δ` is the
  challenge whose absence was the 2022 BP++ draft's soundness hole; its
  position in this schedule is regression-tested.)
- Every squeeze MUST reduce the full 32-byte digest mod `n` (≈256-bit
  challenge space). MUST NOT truncate to 160 bits. The challenge space is
  the denominator of every cheating probability in §4; truncation silently
  caps the whole construction's soundness at `2^-160`. (The sibling
  backend's `bcl24` `CHALLENGE_BITS = 160` is a class-group artifact with
  no EC analogue — routing any squeeze through it is a soundness bug.)
- **One shared sigma challenge.** The final `x` is the verifier challenge
  for every *sigma* sub-protocol — every carry residual and the linking
  sigma. This is the parallel AND-composition of Σ-protocols under a
  common challenge (§4.4); the extractor rewinds the one `x` and runs each
  component extractor on its slice. The BP++ segment has its own
  extraction story (§4.1), glued to the sigma layer through the shared
  Pedersen commitments, not through a shared challenge.

## 3. Absorption checklist

> **Note.** Frozen order. Sole canonical definition of the transcript — on any
> discrepancy with code or other docs, **this list wins**. Restatements elsewhere
> are informative only.

Three deliberate exclusions, before the list:

- The transcript `protocol_id` (item 1) is the exact byte string
  `b"ve-capsule.ec-segve.secp256k1.v1"` and subsumes the backend identity. There
  is no separate "backend-id" transcript field; the framework's
  `PackageConstructionId`/`RecoveryKeyKind` label `ec-segve-secp256k1-v1`
  is a wire tag, not a transcript field.
- `2^ℓ` is not absorbed as a separate field — it equals `d^D` and is fully
  determined by the absorbed `d` (item 6) and `D` (item 7).
- **`T` is not absorbed.** The proof binds the fragment commitment `C`
  (item 11); `T` is bound externally by the framework's `C == T` gate (§5).
  Absorbing `T` would conflate the offline proof statement with the
  framework target the proof is later checked against.

Absorb in exactly this order, every item length-prefixed, before squeezing
`x`. The prover and every verifier reconstruct this identical sequence; an
omitted or reordered item is a soundness break.

**(i) Protocol / version header**

1. `protocol_id`  = `b"ve-capsule.ec-segve.secp256k1.v1"`
2. `statement_version` = `u8` (bump on any wire/statement change)
3. `challenge_width`   = `u16` = `256` (pins the §2 full-reduce decision
   into the transcript)

**(ii) Parameters** (frozen consts, still absorbed so a parameter swap
forks the challenge)

4. `ℓ`  (`u16`)
5. `L`  (`u16`)
6. `d`  (`u16`) — the BP++ digit base (16)
7. `D`  (`u16`) — digits per limb (6)
8. `n`  (32-byte BE group order)

**(iii) Caller context** (application `Context`)

9. `context.domain()`        (UTF-8 bytes)
10. `context.binding_bytes()` (canonical payload: session_id, epoch≥1,
    recipient_id, consensus_digest)

**(iv) Statement** (all public points, re-serialized canonical SEC1)

11. `C`   — fragment commitment `m·G`
12. `pk`  — recovery public key. The verifier MUST have already rejected
    `pk = identity` (and any off-curve / non-canonical encoding) at decode /
    `seal` / `verify` (§5): with `pk = O`, every `D_k = v_k·G` is public
    and limb secrecy collapses. Identity `pk` is a hard reject, not a
    transcript fork.
13. `H`   — the NUMS generator, recomputed locally (§0) and re-serialized;
    never a wire value. Absorbed so that an implementation drift in `H`
    forks the challenge; binding rests on the local recompute + KAT, not on
    absorption.
13a. `generators_digest` — the 32-byte digest pinning the BP++ vector
    generators `g⃗`/`h⃗` (§0). Same rationale as item 13: drift forks the
    challenge; binding rests on local derivation + KAT.
14. limb weights `2^{ℓk}` for `k ∈ [0, L)` — as the count, then each weight
    (32-byte BE scalar)
15. ElGamal list: count `L`, then for each `k`: `E_k` ‖ `D_k`
16. Commitment list: count `L`, then for each `k`: `Com_k`
17. Complement-commitment list: count `L`, then for each `k`: `Com̄_k`
18. Carry-commitment list: count `L−1`, then each `ComC_k = c_k·G + g_k·H`
    for `k ∈ [0, L−1)` (the commitments to the boolean carries; `c_{-1}=0`
    and `c_{L-1}=0` are constants, not committed). With this item the
    **entire statement** — every value the BP++ circuit will range-bound —
    is absorbed before any challenge exists.

**(v) BP++ segment** (multi-squeeze; every prover flight absorbed before
its challenge, §2). The flight order is the reciprocal-form protocol of
[BP++ §6.2–6.3]: the digit and multiplicity commitments (`C_L`, `C_O`)
MUST precede `α` — the reciprocal argument's soundness (Lemma 1) requires
the multiset and its declared multiplicities to be fixed before the
challenge that probes them — and the reciprocal commitment `C_R` can only
follow `α` (reciprocals are functions of it).

19. Initial-witness flight: `C_L` (digits in its norm slot, multiplicities
    in its linear slot), then `C_O` (blinding; its witness image is empty
    in the shared-multiplicity layout) — two points, that order.
    → squeeze `alpha`. All of items 1–18 plus this flight precede it, so
    `α` binds the full statement, context, digits, and multiplicities.
20. Reciprocal flight: `C_R` (one point — the reciprocal vector `w⃗_P(α)`
    in its norm slot). → squeeze `rho`, `lambda`, `beta`, `delta` (four
    sequential ratcheted squeezes, §2).
21. Blinding flight: `C_S` (one point). → squeeze `tau`.
22. WNLA fold rounds, `i = 1…F` (`F = 6` at the frozen shape, §4.1): per
    round absorb `absorb_u16(i)`, `absorb_point(X_i)`, `absorb_point(R_i)`
    → squeeze `gamma ‖ BE16(i)`. The round index is the verifier's own
    loop counter. The final clear vectors `l⃗, n⃗` are wire payload checked
    by the WNLA base-case equation — computed before `x` is squeezed but,
    like all responses, never absorbed.
23. (Reserved — renumbering guard so the sigma items keep their indices.)

**(vi) Sigma announcements**

24. Carry-residual Schnorr announcements: for each limb equation
    `k ∈ [0, L)`, the announcement `A^R_k` proving the public residual
    `R_k` (§4.2) is a pure `H`-multiple.
25. Linking-sigma announcements: three lists, each absorbed as a raw 4-byte
    BE count (`L`) then its elements in limb order — `{A^E_k}`, then
    `{A^D_k}`, then `{A^{Com}_k}` (legs a/b) — followed by the single
    aggregate point `A^C` (leg c, no count). See §4.3 for the exact points.

**THEN** squeeze `x` (label `sigma.x` — the final squeeze, §2). All sigma
responses are computed from `x`; no further absorption.

> **Implementation note:** expose typed `absorb_*` methods (no raw byte
> escape hatch on the safe path), `squeeze(&mut self, label) -> Scalar` for
> the interior challenges, and a terminal `finalize(self, label) -> Scalar`
> that consumes the transcript for `sigma.x` — the type system enforces the
> final-squeeze rule. A KAT pins the ratchet derivation byte-exactly (a
> fixed challenge vector over a fixed absorption/label sequence —
> `ratchet_vector_kat`); negative KATs prove that swapping any two absorbed
> items, swapping any two squeeze labels, or reordering the `ρ→λ→β→δ`
> sequence changes every downstream challenge; and the protocol-level
> Frozen-Heart matrix proves every flight absorption and label is
> load-bearing end to end. (The full statement→challenge pipeline is pinned
> indirectly: params id, generators digest, proof wire bytes, and the
> Partial vector are all golden.)

## 4. Concrete sub-protocol equations + soundness

> **Normative.** Seal-proof equations. BP++ algebra by reference ([BP++]+[CS-BPPP]); sigma layer fully stated. Impl: `range_circuit`, `norm_arg`, `carry`, `linking`, `assembly`.

Two proof systems share the transcript. The **BP++ aggregate** (§4.1) has
its own multi-round challenges and its own (multi-round) extraction story.
The **sigma layer** (§4.2 carry residuals, §4.3 linking) consists of
Σ-protocols sharing the single final challenge `x` (§2), each
**2-special-sound** (a witness extractor exists from two accepting
transcripts with the same first move and distinct `x ≠ x'`) and **HVZK** (a
simulator produces accepting transcripts given `x`, without the witness).
§4.4 composes the two systems through the shared Pedersen commitments.
[Mau09]-style proofs of a homomorphism preimage are written "prove
knowledge of `w` s.t. `Y = φ(w)`": announcement `A = φ(nonce)`, response
`z = nonce + x·w`, check `φ(z) = A + x·Y`.

### 4.1 Aggregated BP++ reciprocal range proof (all limbs + carry booleanity)

**Goal:** `v_k ∈ [0, 2^ℓ)` for every value limb, the same for every
complement limb, and `c_k ∈ {0, 1}` for every carry — all in **one**
aggregated proof over the already-absorbed Pedersen commitments
`{Com_k, Com̄_k, ComC_k}`. The normative protocol is the BP++ reciprocal
range proof [BP++, §5–§6] instantiated at the frozen shape below, with the
extraction lemma taken from the CypherStack-corrected statement [CS-BPPP]
(the eprint's Lemma 5 is invalid as written; the corrected proof is
normative). This section pins the statement, witness layout, flights, and
verifier obligations; the full constraint algebra lives in the cited
sources and the implementation, KAT-pinned.

**Statement and witness layout.** The circuit takes `W = 2L + (L−1) = 32`
committed values in frozen order: value limbs `k = 0…10`, complement limbs
`k = 0…10`, carries `k = 0…9`. Each value `w` opens its commitment over
`(G, H)` — `V_w = val_w·G + blind_w·H` — which is exactly the capsule's
Pedersen form (`h⃗[0] = H`, §0), so the capsule commitments enter the BP++
statement directly, with no re-commitment. Two digit groups:

- **base-16 group:** each limb value decomposes into `D = 6` digits;
  `22·6 = 132` digits total, with 15 explicit shared multiplicity slots
  (symbols `1..=15`) counting occurrences across all 132 digits — the
  zero-symbol multiplicity is implicit in Eq. 75's `X`-pole term;
- **base-2 group:** each carry is its own single digit; `10` digits, one
  explicit multiplicity slot (symbol `1`), zero implicit.

The 15 + 1 explicit slots are the 16-entry shared multiplicity vector that
fills `l⃗_L` exactly (the frozen-shape table).

Per group, the circuit enforces (i) digit recomposition
(`val_w = Σ_j d_{w,j}·16^j`, resp. `c_k = d_k`), (ii) the reciprocal
products `r_{w,j}·(α + d_{w,j}) = 1`, and (iii) the pole equation
`Σ digits 1/(α + d) = Σ_s m_s/(α + s)` over the group's symbol set. By the
reciprocal set-membership argument (preliminaries), a random `α` agreeing
on both sides forces every digit into the symbol set; recomposition then
gives `val_w < 16^6 = 2^24` exactly (no over-cover — the BSGS window
invariant), and `c_k ∈ {0, 1}`.

**Flights and challenges (normative order; absorption is §3 items 19–22).**
The reciprocal-form protocol [BP++ §6.3]:

1. Statement absorbed (§3 items 1–18). Prover commits the initial witness:
   `C_L` (digits `w⃗_D` in its norm slot, the 16 shared multiplicities in
   its linear slot) and `C_O` (blinding only — the shared-multiplicity
   layout leaves its witness image empty) → squeeze `α`. The digits and
   multiplicities are therefore bound BEFORE the challenge that probes
   them — the Lemma-1 soundness precondition.
2. Prover computes the reciprocals `w⃗_P(α)` (erroring on a vanishing
   denominator, §"Verifier obligations") and commits `C_R` → squeeze
   `ρ, λ, β, δ` (sequential). `μ := ρ²` is derived, not squeezed.
3. Blinding flight `C_S` → squeeze `τ`.
4. WNLA fold rounds `i = 1…F = 6`: absorb `(X_i, R_i)` → squeeze `γ_i`;
   base case ships the residual `l⃗, n⃗` in the clear.

**Frozen-shape table (unit-3 derivation from [BP++ §5.3, §6.3–6.4.2],
shared-multiplicity layout).**

| Constant | Value | Derivation |
|---|---|---|
| `k` (committed values) | 32 | 11 value limbs ‖ 11 complement limbs ‖ 10 carries (frozen order) |
| `N_v` (value-vector width) | 16 | slot 0 = the committed value; slots 1–15 forced zero by all-zero `W_l` rows; width chosen so the 16 shared multiplicities (15 base-16 symbols + 1 base-2 symbol; the zero-symbol multiplicity is implicit per Eq. 75) fill `l⃗_L` exactly |
| `N_p` (digits = poles) | 142 | `22·6 + 10·1` |
| `N'_m` (mult gates / n-side) | 142 | one reciprocal product per digit; original circuit has zero mult rows |
| `N_O` (w_O width) | 16 | the shared multiplicities |
| `N'_w` (witness width) | 300 | `142 + 142 + 16` (`w⃗' = w⃗_D ‖ w⃗_P ‖ w⃗_O`) |
| `N'_l` (linear rows) | 514 | 512 `v`-aligned rows (row `16i`: digit recomposition of value `i`; rows `16i+j, j≥1`: zero-forcing of unused `v` slots) + 2 pole-equation rows (base-16, base-2) |
| flags | `f_l = 1, f_m = 0` | Eq. 74 requires `f_l = 1`; numerators are the constant 1 (`W_n = 0, a⃗_n = 1⃗`) so `f_m = 0` |
| l-side width | 23 → padded 32 | `N_v + 7`; zero-padded for the WNLA |
| n-side width | 142 → padded 256 | `N'_m`; zero-padded for the WNLA |
| `F` (fold rounds) | 6 | `(32, 256) → (1, 4)` under the `l+n < 6` rule |
| residual `l⃗`/`n⃗` | 1 / 4 | ditto |
| **wire artifact** | **16 points + 5 scalars = 688 B** | `C_L, C_O, C_R, C_S` + `6·(X_i, R_i)` + residuals |

Binding requirements, independent of the numbers: every count is a
compile-time constant (the wire format has **zero length fields**, §7);
the WNLA pads with zeros to the frozen lengths and the verifier's
fixed-shape checks pin the fold count and residual lengths — the wire has
NO padding coordinates to inject into. Mass a prover hides on a padded
generator slot inside its own flight commitments is **benign by
construction**: the constraint vector `c⃗(τ)` is zero on padded slots and
no constraint row reads them, so such mass is indistinguishable from
blinding and changes nothing about the proven statement (demonstrated by
the `padded_slot_mass_is_benign_blinding` test); `F` and the residual
lengths derive from the padded widths, never the wire. The generator counts (§0)
freeze with this table: `g⃗` = 256, `h⃗` = 32.

**Verifier obligations.**

- Recompute the circuit weights from `α` and the frozen shape (never from
  the wire), re-derive every challenge by §2/§3, and run the BP++
  verification equation — one multi-scalar product over the statement
  commitments, flights, generator vectors, and clear residuals.
- Decode every wire scalar canonical `< n` and every point strict-SEC1
  (§1); reject **identity** for every flight point (`C_L`, `C_O`, `C_R`,
  `C_S`, `X_i`, `R_i`) — an identity flight is always degenerate
  (it is the simulator's freebie, never an honest prover's output).
- **Pole handling:** if any verifier-side public denominator `(α + s)`
  vanishes, reject (do not panic). Prover-side, a vanishing `(α + d)` is a
  negligible-probability event (`α` is squeezed after the digits are
  bound); the prover errors out rather than resampling — there is nothing
  sound to resample in a Fiat–Shamir proof.
- **No early state:** all checks are post-decode; the single decode door
  (§7) has already enforced shape and canonicality.

**Soundness.** Knowledge-soundness of the BP++ argument is [BP++ Thm. 3 /
Thm. 4] with the CypherStack-corrected round-extraction lemma [CS-BPPP],
under the discrete-log relation assumption over `(G, H, g⃗, h⃗)` (NUMS,
§0). Fiat–Shamir for a multi-round protocol carries the standard
multi-round caveat: extraction is via a tree of transcripts (one fork per
challenge round), with the corresponding security loss; the ≈256-bit
challenge space per squeeze (§2) keeps the concrete bound far beyond
reach. The `δ` challenge is load-bearing — its omission (or any absorption
gap before it) re-opens the 2022 draft's soundness hole; §2's schedule and
the regression tests pin it.

**Zero-knowledge.** The BP++ blinding flights (`C_S` plus the blinding
structure inside `C_L/C_R/C_O` and the WNLA) give honest-verifier ZK per
[BP++ Thm. 2, SHVZK as corrected]; nothing about the digits, reciprocals,
or multiplicities leaks beyond the statement. The commitments themselves
remain hiding exactly as before.

**What binds this to the rest of the capsule:** the BP++ statement
commitments ARE the capsule's `Com_k / Com̄_k / ComC_k` (same points, same
generators). Pedersen binding makes each committed value unique, so the
values BP++ range-bounds are the same values the carry chain (§4.2)
telescopes and the linking sigma (§4.3) connects to the ciphertexts and
`C`. No proof re-asserts another's statement; the glue is the shared
commitments plus binding.

### 4.2 Exact-integer carry chain (`m + m̄ = n − 1`)

**Goal:** the integer `M = Σ_k v_k·2^{ℓk}`, reassembled from the limbs that
§4.1 just bounded, satisfies `M ≤ n − 1` — so `M` is the unique canonical
representative of its residue class, and "what the package decrypts to" is
`m` itself, not merely something congruent to it.

The range proofs alone do not suffice here: §4.1 bounds each limb, which
only bounds `M < 2^{L·ℓ} = 2^264` — roughly `2^8·n`. A dishonest prover
still has room to encode `M = m + n`: every limb in range, and the curve
cannot object, because curve equations see scalars mod `n` — the linking
check (§4.3) constrains `M·G = C`, and `(m + n)·G = m·G`. Any purely
mod-`n` relation is blind to the wraparound. The carry chain removes
exactly this freedom by proving an identity **over the integers**:
`m + m̄ = n − 1` with both sides built from nonnegative limbs, which forces
`M ≤ n − 1`.

The proof is schoolbook addition, carried out in commitments. Add `m` and
`m̄` column by column in base `2^ℓ`; each column must produce the matching
digit of the known constant `n − 1`, plus a carry bit into the next
column:

```
v_k + v̄_k + c_{k−1} = (n−1)_k + c_k·2^ℓ      for all k ∈ [0, L)
```

with `c_k ∈ {0,1}`, no carry into the lowest column (`c_{-1} = 0`), no
carry out of the highest (`c_{L-1} = 0`), and `(n−1)_k` the public `k`-th
base-`2^ℓ` digit of `n − 1`. The whole chain is proven over the
additively-homomorphic Pedersen commitments, column by column.

**Carry commitments + booleanity.** `ComC_k = c_k·G + g_k·H`
(`g_k ←$ Z_n`) for `k ∈ [0, L−1)` (item 18). Each `c_k ∈ {0,1}` is proven
by the §4.1 aggregate's **base-2 digit group** — the carry commitments are
statement values of the BP++ circuit, range-bound to `[0, 2)`. Define
`ComC_{-1} := O` and `ComC_{L-1} := O` — the constant zero carries; `O` is
the identity, with opening `(0, 0)`.

**Per-limb `H`-residual Schnorr.** From public data alone, the verifier
forms the commitment-level version of column `k`'s equation:

```
R_k := Com_k + Com̄_k + ComC_{k−1} − (n−1)_k·G − 2^ℓ·ComC_k
```

Expanding with the openings,
`R_k = (v_k + v̄_k + c_{k−1} − (n−1)_k − 2^ℓ·c_k)·G + (s_k + s̄_k + g_{k−1} − 2^ℓ·g_k)·H`.
If column `k`'s equation holds on the values, the `G`-component vanishes
and `R_k` is a pure `H`-multiple — its `H`-component is blinding
bookkeeping, known to the prover. So the prover proves exactly
`R_k ∈ ⟨H⟩` — `∃ w_k : R_k = w_k·H` — by a Schnorr proof of dlog-w.r.t.-`H`
(announcement `A^R_k = ρ_k·H`, `ρ_k ←$ Z_n`, item 24; response
`z^R_k = ρ_k + x·w_k`; check `z^R_k·H == A^R_k + x·R_k`), where the honest
witness is `w_k = s_k + s̄_k + g_{k−1} − 2^ℓ·g_k`. (The nonce symbol `ρ_k`
here is the sigma layer's own; it is unrelated to the BP++ challenge `ρ`.)

The soundness chain, in three steps:

- **Special soundness ⇒ the `G`-coefficient is 0 mod `n`.** Extraction
  yields `w_k` with `R_k = w_k·H`. Set against the expansion above,
  `(v_k + v̄_k + c_{k−1} − (n−1)_k − 2^ℓ·c_k)·G = (w_k − [s_k+s̄_k+g_{k−1}−2^ℓ·g_k])·H`.
  A nonzero `G`-coefficient here would express `G` as a known multiple of
  `H` — computing the dlog of `H`, equivalently breaking Pedersen binding.
  So `v_k + v̄_k + c_{k−1} − (n−1)_k − 2^ℓ·c_k ≡ 0 (mod n)`.
- **Mod-`n` ⇒ integer (the crux).** From §4.1, `v_k, v̄_k ∈ [0, 2^ℓ)`; from
  the §4.1 base-2 digit group, `c_{k−1}, c_k ∈ {0,1}`. So the left side
  `v_k + v̄_k + c_{k−1}` lies in `[0, 2^{ℓ+1}−1]`, the right side
  `(n−1)_k + 2^ℓ·c_k` lies in `[0, 2^{ℓ+1}−1]`, and their integer
  difference lies in `(−2^{ℓ+1}, 2^{ℓ+1})`. Since `2^{ℓ+1} = 2^25 ≪ n`, the
  only value in that window that is `≡ 0 (mod n)` is `0` itself. Each
  column equation therefore holds **over `Z`**, not just `Z_n`. This step
  is where the proven ranges pay for themselves: without them the values
  could sit anywhere mod `n` and the window argument would say nothing.
- **Telescoping.** Multiply column `k` by `2^{ℓk}` and sum over `k`. The
  value sums give `Σ_k 2^{ℓk}·v_k = M`, `Σ_k 2^{ℓk}·v̄_k = M̄`, and
  `Σ_k 2^{ℓk}·(n−1)_k = n−1`. The carry terms cancel in pairs:
  `Σ_k 2^{ℓk}·c_{k−1} = Σ_k 2^{ℓ(k+1)}·c_k` (reindex, using `c_{-1}=0`)
  against `Σ_k 2^{ℓk}·2^ℓ·c_k = Σ_k 2^{ℓ(k+1)}·c_k`, with `c_{L-1}=0`
  killing the would-be overflow term at the top. What remains is
  `M + M̄ = n − 1` over `Z`; with `M, M̄ ≥ 0` this forces
  `M ≤ n − 1 < n`. ∎
- **HVZK.** The carry commitments are hiding (fresh `g_k`); booleanity is
  part of the §4.1 zero-knowledge aggregate, and the residual Schnorrs
  simulate exactly as in §4.3.

### 4.3 Linking sigma — one shared-response multi-representation proof

**Goal (the cross-binding).** After §4.1–§4.2, the *commitments* contain
in-range limbs assembling to a canonical value — but nothing yet says the
*ciphertexts* contain those same limbs, nor that the limbs assemble to the
discrete log of `C` specifically. The linking sigma proves there is a
single witness `{v_k, r_k, s_k}_k` simultaneously satisfying

- (a) `E_k = r_k·G` and `D_k = v_k·G + r_k·pk` — the ElGamal pairs are
  well-formed, with the mask handle and masked value built from the same
  `r_k`;
- (b) `Com_k = v_k·G + s_k·H` — the same `v_k` sits in the Pedersen
  commitment;
- (c) `(Σ_k 2^{ℓk}·v_k)·G = C` — the limbs open the fragment commitment.

Independent Schnorr proofs of (a), (b), (c) under a shared `x` would NOT
suffice: each leg would bind its own opening, but no check would ever
compare the `v_k` inside `D_k` with the `v_k` inside `Com_k`. The binding
mechanism is **response reuse**: one response scalar `z_{v,k}` for `v_k`,
checked in every equation where `v_k` appears (and one `z_{r,k}` everywhere
`r_k` appears). A response is a linear function `nonce + x·secret` of
exactly one secret; requiring a single response to satisfy several
verification equations forces that secret to be the same in each. This is
a Maurer [Mau09] generalized Σ-protocol for the joint homomorphism.

- **Prover.** Per proof, draw fresh entropy and derive per-limb nonzero
  nonces `α_k` (for `v_k`), `β_k` (for `r_k`), and `γ_k` (for `s_k`) under
  `ve-capsule.linking-sigma.nonce.v1` from that entropy, the relevant
  witness scalar, the limb index, and a binding to the capsule context plus
  every public statement/range/carry item that can affect `sigma.x` before
  linking announcements are generated. This is distributionally a fresh nonce
  draw under healthy entropy, while repeated RNG byte streams do not repeat the
  same Schnorr nonce across distinct challenges. Announcements (item 25):
  - `A^E_k := β_k·G`             (leg a, `E_k`)
  - `A^D_k := α_k·G + β_k·pk`    (leg a, `D_k`)
  - `A^{Com}_k := α_k·G + γ_k·H` (leg b)
  - `A^C := (Σ_k 2^{ℓk}·α_k)·G`  (leg c — one aggregate point over all limbs)

  After `x`: responses `z_{v,k} := α_k + x·v_k`, `z_{r,k} := β_k + x·r_k`,
  `z_{s,k} := γ_k + x·s_k`.
- **Verifier.** Decode responses canonical `< n` (§1). Check, per limb `k`:
  - `z_{r,k}·G == A^E_k + x·E_k`                   (binds `r_k` in `E_k`)
  - `z_{v,k}·G + z_{r,k}·pk == A^D_k + x·D_k`      (same `z_{v,k}`, `z_{r,k}`)
  - `z_{v,k}·G + z_{s,k}·H == A^{Com}_k + x·Com_k` (same `z_{v,k}`)

  and once: `(Σ_k 2^{ℓk}·z_{v,k})·G == A^C + x·C`  (same `z_{v,k}`).

  The reuse of `z_{v,k}` across the `D_k`, `Com_k`, and weighted-`C` checks
  is what forces one `v_k`; the reuse of `z_{r,k}` across the `E_k` and
  `D_k` checks forces one `r_k` and makes leg (a)'s `E_k = r_k·G` mandatory
  and bound.
- **Special soundness / extraction.** From two transcripts sharing
  announcements with `x ≠ x'`: `v_k := (z_{v,k} − z'_{v,k})/(x − x')`,
  `r_k := (z_{r,k} − z'_{r,k})/(x − x')`,
  `s_k := (z_{s,k} − z'_{s,k})/(x − x')`. Substituting back into the four
  checks shows the extracted `{v_k, r_k, s_k}` satisfy (a), (b), (c) with
  the same `v_k`/`r_k` in every equation — the same difference quotient
  defines them wherever they appear. In particular leg (a) gives
  `E_k = r_k·G` exactly, with the same `r_k` as in `D_k`, so decryption is
  mechanical: `D_k − sk·E_k = v_k·G + r_k·pk − sk·r_k·G = v_k·G`. This is
  the check that defeats the malformed-`E_k` ransom — an `E_k` built from
  different randomness than `D_k` (or from none) would make
  `D_k − sk·E_k` a point with no small dlog, the bounded BSGS search would
  come up empty, and a "verified" package would be undecryptable.
- **Cross-binding to §4.1/§4.2 (through the shared public `Com_k`).** The
  range proof (§4.1) and the carry chain (§4.2) prove statements about the
  same public `Com_k` that leg (b) opens. Pedersen binding makes the
  committed value unique, so the `v_k` extracted here equals the `v_k` the
  range proof bounds to `[0, 2^ℓ)` and the carry chain assembles to
  `M ≤ n−1`. The sub-proofs are glued by the shared commitment plus
  binding — no proof re-asserts another's statement.
- **HVZK.** Draw all responses `←$ Z_n` and back-compute each announcement
  from its check equation (e.g. `A^E_k := z_{r,k}·G − x·E_k`). Perfect
  HVZK; no `v_k, r_k, s_k` leaks. Secrecy of the limb values additionally
  rests on the ElGamal ciphertexts being IND-CPA (DDH) and `Com_k` hiding.

### 4.4 Composition — knowledge-soundness, zero-knowledge, secrecy

**Claims.** Two load-bearing properties are claimed, plus one structural
one:

- **(K) Knowledge-soundness (ROM):** whoever produces an accepting package
  *knows* a witness, and the witness's structure forces the package to
  decrypt to the unique `m ∈ [0, n)` with `m·G = C`. This is the ransom
  defense.
- **(Z) Zero-knowledge (ROM):** the package can be simulated without the
  witness; combined with ElGamal IND-CPA, it leaks nothing about `m` beyond
  the already-public `C`. This is secrecy.
- **Statement binding:** the challenge hashes the entire statement and
  context, so an accepting proof cannot be transplanted onto a different
  statement, session, or protocol.

Deliberately not claimed: simulation-sound extractability / UC composition
(not needed for the above — see below), constant-time execution (see the
execution-environment paragraph), and any secrecy against the encryptor's
own misbehavior or for guessable `m` (see the caveats).

**Composition.** Two proof systems compose over one transcript and one set
of statement commitments:

- The **sigma layer** (§4.2 residuals + §4.3 linking) is a parallel
  AND-composition of 2-special-sound HVZK Σ-protocols under the single
  final challenge `x`: the simulator runs every component simulator on the
  same `x`; the extractor rewinds the one `x` (to `x' ≠ x`) and runs every
  component extractor on its slice. One rewind yields the sigma witness
  `{v_k, v̄_k, r_k, s_k, s̄_k}` and the residual witnesses `{w_k}`.
- The **BP++ aggregate** (§4.1) is knowledge-sound on its own terms
  (multi-round tree extraction, [BP++]+[CS-BPPP]), yielding openings of
  every `Com_k / Com̄_k / ComC_k` with in-range values.
- The **glue is Pedersen binding on the shared commitments**: both
  extractions open the *same* points over the *same* `(G, H)`, so the
  values agree — the limbs BP++ bounds are the limbs the residuals
  telescope and the linking sigma connects to `{E_k, D_k}` and `C`. The
  transcript chaining (§2) additionally binds the two systems'
  challenges into one ordered schedule, so neither proof can be generated
  against a different statement or grafted from another session.

**(Z) and (K), precisely.** (Z): the FS proof is zero-knowledge in the ROM.
The BP++ simulator ([BP++ SHVZK, as corrected]) and the §4.2/§4.3
simulators run on programmed challenges; the composite simulator programs
the random oracle at each squeeze point of §2's schedule; high-entropy
challenges give negligible programming-collision error. (K): the proof is
knowledge-sound in the ROM. The sigma layer extracts by the general
forking lemma [PS00] at the final squeeze; the BP++ layer extracts by the
multi-round generalization (a transcript tree with one fork level per
challenge round, the standard multi-round FS analysis), using the
CypherStack-corrected round-extraction lemma. Both extractions are against
the same absorbed statement; Pedersen binding merges the witnesses. The
multi-round tree extractor's looser concrete bound is absorbed by the
≈256-bit per-squeeze challenge space (§2).

**Knowledge-soundness ⇒ the ransom defense.** A single accepting proof's
extracted witness satisfies, simultaneously: every digit in its symbol set,
hence `v_k, v̄_k ∈ [0, 2^ℓ)` exactly (§4.1, `16^6 = 2^24`); each
`c_k ∈ {0,1}` (§4.1 base-2 group); the integer column equations, hence
(telescoping) `M ≤ n−1`; `E_k = r_k·G` and `D_k` well-formed; `Com_k`
opening to the same `v_k`; and `(Σ_k 2^{ℓk}·v_k)·G = C`. Therefore the
ciphertexts `{E_k, D_k}` decrypt — exactly, because `r_k` is bound — to
limbs that reassemble to the unique `m ∈ [0, n)` with `m·G = C`. A prover
who plants a package decrypting to anything else must break the BP++
argument (DL-relation assumption + ROM), a Σ-protocol's soundness
(probability `≤ 1/n` per attempt), or Pedersen binding (dlog of `H`).

**Non-malleability via statement binding (simulation-soundness is NOT
claimed).** v1 deliberately does not rely on simulation-sound
extractability, and does not invoke [FKMV12]'s quasi-unique-responses
theorem — the proof systems here are not claimed to have unique responses,
and claiming it would be an overclaim. v1 needs and proves only (Z) and (K)
above, and neither consumes simulation-soundness. The non-malleability v1
actually requires — a proof cannot be mauled onto a *different* statement,
nor replayed across sessions or protocols — is delivered by strong-FS
statement binding: every challenge in §2's schedule transitively hashes the
domain (§1), `statement_version`, the absorbed `Context`
(session/epoch/recipient/consensus), and the entire statement (`C`, `pk`,
every `E_k/D_k/Com_k/Com̄_k/ComC_k`, the generator digests) — the first
squeeze `α` already binds all of it, and the ratchet (§2) carries that
binding into every later challenge including `x`. Any change to statement
or ceremony yields a different challenge vector and the mauled proof fails
its checks. A full simulation-sound / UC treatment is explicitly out of v1
scope; no guarantee claimed here needs it.

**Secrecy (HONEST-ENCRYPTOR scope).** Against a coalition without `sk` —
the server, sub-threshold helpers, any verifier: by (Z) the proof is
simulatable without the witness, so it leaks nothing beyond the public
statement; the ciphertexts `{E_k = r_k·G, D_k = v_k·G + r_k·pk}` are
IND-CPA under DDH on secp256k1 (standard ElGamal); and
`Com_k`/`Com̄_k`/`ComC_k` are perfectly hiding (fresh blindings,
unknown-dlog `H`), with the BP++ vector commitments and flights hiding /
SHVZK per §4.1. The precise claim: the adversary's view adds **nothing
beyond the public discrete-log instance `C = m·G`** (equivalently `T`,
already public as the certified verifying share). Recovering `m` from the
full VE package is no easier than solving the dlog of the already-public
`C` — the encryption and proof contribute zero additional leakage. This is
the right claim because `C`/`T` are public by construction; it is NOT the
stronger "semantically hidden `m`". The consequence: for a full-entropy
scalar share (the FROST use case, `m` uniform in
`[0, n)`) this is ideal — dlog is hard. For a low-entropy or guessable `m`,
`C = m·G` is already publicly guess-checkable (test `m?·G == C`)
independent of this scheme; ec-segve neither adds nor removes that
exposure. Callers MUST NOT use it to escrow low-entropy secrets expecting
secrecy.

**Honest-encryptor caveat (scoped, not a gap).** The VE proof binds
*validity*; it cannot force *hiding randomness*. A malicious encryptor may
set `s_k = 0`, making `Com_k = v_k·G` — a BSGS-recoverable point that
self-leaks its own `m`. This grants the encryptor nothing it lacks (it
holds `m`), so it is explicitly outside the secrecy game: secrecy is
claimed only against parties without `sk`, under honest provisioning. This
paragraph and the one above are the IND-CPA + NIZK reduction the design doc
references.

**Retired prototype scans.** An earlier hardening pass grew a family of
verifier-side *relation scans* — publicly-enumerable / small-coefficient
searches over the Pedersen commitments, carry commitments, linking
responses, OR sub-challenges/responses, and proof announcements — each
attempting to block a malicious encryptor from choosing low-entropy
randomness that self-leaks its own `m`. Those are retired in v1: they are
not §5 gates, under the honest-encryptor scope they defend nothing the
encryptor could not already do (it holds `m`), and they cost the verifier
seconds per package (e.g. a cubic `C(L,3)·32³` linking-response scan).

**Retained mask gates (as built).** What v1 keeps is the check family over
the ElGamal masks `{E_k}` themselves, because a structurally degenerate
mask set leaks limb information to **every observer** — the same class as
`E_k = identity` — guarding an *honest* recipient's secrecy against a
malformed ciphertext rather than encryptor self-leak:

- `E_k = identity` (`r_k = 0`, an unmasked limb — `D_k = v_k·G` is then
  public and BSGS-recoverable): rejected at wire decode
  (`Proof::from_canonical_bytes`), at `seal`, and at `verify`.
- A duplicate (`E_i = E_j`) or inverse (`E_i = −E_j`) mask pair, and any
  coefficient-`≤ 2` linear relation among the masks (subsuming coefficient-one
  signed subsets): each
  makes a small public combination `Σ cᵢ·Dᵢ` cancel the recovery-key term
  and publish a bounded relation between limb plaintexts
  (BSGS-searchable). The simplest instance: `E_i = E_j` means `r_i = r_j`,
  so `D_i − D_j = (v_i − v_j)·G` — a public point with a small dlog.
  The retained gate also rejects bounded public-`G` mask openings and pairwise
  low-coefficient public-`G` offsets: if `Σ cᵢ·Eᵢ = q·G` for a small public
  `q`, then `Σ cᵢ·Dᵢ − q·Y* = (Σ cᵢ·vᵢ)·G` is the same bounded limb leak.
  Rejected at `seal` and `verify` (a split-half meet-in-the-middle over the
  `L = 11` masks with coefficients `≤ 2`; bounded cost, unlike the retired
  cubic scans).
- For `Case` bundles, per-piece proof validity is not enough. A malicious
  helper can choose its own proof-valid piece with masks publicly related to an
  honest helper's masks; then the cross-piece `D` combination cancels the same
  `Y*` term and leaks a bounded relation over the honest helper's limb. The
  full `Case::verify` and stripped `StrippedCase::bind` paths therefore reject
  public scalar and pairwise public-`G` offset mask relations across different
  pieces through the public-enumerability window, global unit-coefficient
  relations through support six, and coefficient-two relations through support
  four under the admitted six-piece profile.
- Proof-backed `Case` bundles also retain BP++ statement-commitment relation
  gates. Per-piece Pedersen commitments are locally valid, but a malicious
  helper can reuse or split the `H` blinding from an honest helper's statement
  commitment and make a public cross-piece combination land on a bounded
  `G` multiple. `Case::verify` therefore rejects pairwise public-`G`
  commitment relations across all statement slots and exhaustive same-slot
  unit relations under the admitted six-piece profile. Stripped Case cores do not carry BP++ statement commitments, so this
  proof-backed gate has no stripped-core analogue.

**Execution-environment assumption (side channels).** The proofs above are
sound and zero-knowledge at the protocol level; the implementation makes
**no constant-time claims** on the secret-handling paths (`seal`,
`contribute`, `unseal`). Concretely: the BP++ prover (§4.1) computes
reciprocals `1/(α + d)` with variable-time field inversion and walks
digit-valued witness vectors — a fine-grained local observer
(timing / cache / power) of `seal` could recover information about the
digits of `m`; the BSGS recovery walk is variable-time in the recovered
limb values by design (`src/bsgs.rs`). The threat model therefore assumes `seal`
executes on the device that already holds `m`, and `unseal` on the
recipient device entitled to recover it. `verify` touches only public data.

## 5. Verifier-gate → enforcing sub-component (traceability)

> **Normative map.** Each verifier gate → enforcing component and soundness basis.

Every normative verifier gate maps to a proof sub-component or an explicit
verifier check. The as-built traceability check confirms each row has code and a test.

| Verifier gate (design doc) | Enforced by | Soundness basis |
|---|---|---|
| Recovery `pk` authenticity (enrollment-bound, not substituted) | **normative helper gate:** verify `participant_id == H(LOGICAL_ID_DOMAIN, auth_key)` + recovery-pk committed in the threshold-signed reviewed enrollment, BEFORE `seal` | bound recovery key precondition; defeats rogue-key substitution |
| Recovery `pk` well-formed (`≠ identity`, on-curve, canonical SEC1, and not a publicly-enumerable small signed `c·G` multiple nor a small signed `c·H` NUMS multiple) | explicit reject at `from_canonical_bytes` / `seal` / `verify` (`reject_degenerate_recovery_key` + the canonical SEC1 codec) | a `pk` with a public dlog (identity, or small `c·G`) makes every `D_k = v_k·G + r_k·pk` decryptable by any verifier ⇒ secrecy collapse; a `c·H` NUMS multiple has no known dlog ⇒ a package that verifies but is unrecoverable by any signer (availability) |
| Gated recipient/access components are independent enough to enforce the AND policy | explicit component checks in `composite_key`: identity/publicly-enumerable rejection, duplicate access-key rejection, bounded rational scalar relation rejection, low-coefficient two-source relation rejection, unit signed-subset relation rejection, low-target-coefficient unit-subset rejection, and mixed-coefficient signed-subset rejection over the capped roster | related holders must not be able to synthesize another mandatory component's DLEQ-valid opening partial or private scalar from their own secrets |
| Per-limb range `v_k, v̄_k ∈ [0, 2^ℓ)` | §4.1 aggregated BP++ (base-16 group, `16^6 = 2^24`) | reciprocal set membership + digit recomposition, exact, no over-cover; DL-relation + corrected extraction [CS-BPPP] |
| Carry booleanity `c_k ∈ {0,1}` | §4.1 aggregated BP++ (base-2 group) | same aggregate, second digit group |
| Canonical residue / over-`n` (`m ∈ [0,n)`) | §4.2 exact-integer carry chain | telescoped integer identity, `v_k,v̄_k ≥ 0`, boolean `c_k` |
| BP++ flight points non-identity; residual lengths fixed; pole challenges rejected | explicit §4.1 verifier obligations | identity flights are simulator freebies; the wire has no padding coordinates (fixed-shape residuals); vanishing denominators are degenerate |
| ElGamal well-formedness incl. `E_k = r_k·G` | §4.3 leg (a), shared `z_{r,k}` | binds `r_k`; defeats malformed-`E_k` ransom |
| `Com_k` ↔ `D_k` same `v_k` | §4.3 legs (a)+(b), shared `z_{v,k}` | one response forces one `v_k` |
| Ciphertext opens to `C` | §4.3 leg (c), shared `z_{v,k}` | weighted-sum opening, same `v_k` |
| Scalar canonicality (`z_•, l⃗, n⃗ < n`) | explicit decode check (§1) | no `z+n` wire-encoding malleability |
| Structural mask gates: `E_k = identity` (decode/seal/verify); duplicate, inverse, coefficient-`≤ 2` mask relations (including signed subsets), and bounded public-`G` offset mask relations (seal/verify); cross-piece scalar, pairwise public-`G` offset, global unit support-six, and coefficient-two support-four mask relations for `Case` bundles (`Case::verify` / `StrippedCase::bind`) | explicit mask gates (§4.4 "Retained mask gates") | a degenerate mask set publishes bounded limb relations to every observer |
| Cross-piece BP++ statement commitment gates for proof-backed `Case` bundles: pairwise public-`G` relations across all statement slots, plus exhaustive same-slot unit relations under the admitted profile | explicit `Case::verify` Pedersen relation scan (§4.4 "Retained mask gates") | a degenerate cross-piece Pedersen relation cancels `H` and publishes bounded honest-helper limb relations |
| Identity `Com_k`/`C` allowed, one canonical encoding | strict SEC1 codec + canonical absorption (§1) | re-serialize-before-absorb |
| NUMS `H` correct (unknown-dlog basis) | local RFC 9380 recompute + KAT (§0); never wire-read | Pedersen binding rests on `H` |
| FS replay / cross-protocol | §1 domain + §3 item (i) version + §3 item (iii) context | strong-FS statement binding (§4.4) |
| `C == T` (fragment binds to certified share) | **explicit verify-time gate, NOT a proof** | group-element equality `C == T` |

> **`C == T` is intentionally NOT a proof obligation.** The linking sigma
> (§4.3 leg c) only proves the ciphertext opens to `C`; binding `C` to the
> certified target `T` is a separate explicit verifier check in framework
> integration. **Integration contract (type-enforced, single surface):**
> `Capsule::verify` takes the certified target commitment `T` as a
> required argument and returns an opaque `VerifiedCapsule` token only if
> the linking sigma verifies AND the capsule's own canonical `C` equals
> `T`. `Capsule` exposes no bare `commitment()` accessor, so there is no
> second `C` to drift, and no API yields an "accepted" capsule from a bare
> `verify` success without the target comparison — `T` is not optional, so
> a backend `verify` cannot be "accepted" on its own. This forecloses
> (red-team 2026-05-31 #1) an encryptor opening to any `C' = m'·G` and
> passing offline verify — the wrong-`m'` capsule fails the `C == T`
> equality inside `verify`. (Pairs with the possession-certified
> recipient/access-key contract on `seal` that forecloses #2, rogue-key.)

## 6. Contribute-DLEQ transcript

> **Normative.** Contribute-DLEQ transcript (separate from the seal multi-squeeze). Impl: `dleq`, `opening`.

Opening a sealed capsule is itself a small protocol, and it has its own
proof obligation. The opening layer's `Partial` is an authorizer's per-gate
contribution: holding a gate secret `w` with public image `W = w·G`, the
authorizer publishes `W_j = w·E_j` for every segment mask `E_j` in the
capsule. The attached proof is a batched multi-statement **DLEQ**
("discrete-log equality"): the *same* `w` relates `(G, W)` and every
segment pair `(E_j, W_j)`. Without it, a corrupted contribution would
poison recovery silently; with it, a bad `W_j` fails the proof instead of
failing the recovery.

The DLEQ challenge `c` is derived on its **own, separate single-challenge
transcript** using the §1 mechanics (length-prefixed framing, canonical
re-serialized points) and the full 32-byte reduce mod `n` of §2 — squeezed
exactly once. The DLEQ does NOT use the §2 BP++ ratchet, its
`.ratchet`/`.challenge` domains, or its labels; it is deliberately a
separate, single-challenge transcript. Absorb in EXACTLY this order:

1. crate domain `b"ve-capsule.ec-segve.secp256k1.v1"` (the §1 seed)
2. DLEQ sub-domain `b"ve-capsule.contribute-dleq.v1"`
3. `binding` — the canonical contribution binding, one length-prefixed
   field whose payload is itself framed (`push_framed` per component):
   `b"ve-capsule.contribute-binding.v1"` ‖ `digest(core)` ‖ `Y*` ‖
   `g*` ‖ gate ‖ context domain ‖ context binding, where
   `digest(core) = SHA-256(C ‖ (E_0,D_0) ‖ ... ‖ (E_{L-1},D_{L-1}))`
   using the canonical SEC1 limb encoding
4. bases list: raw 4-byte BE count, then `G`, then each segment mask `E_j`
   in segment order
5. images list: same shape — `W = w·G`, then each `W_j = w·E_j`
6. announcements list: same shape — `A_i = k·B_i` in base order (one
   shared nonce `k`, matching the single shared response `z`)

**THEN** squeeze `c`. The verifier checks `z·B_i == A_i + c·P_i` on
**every leg individually** — there is no random-linear-combination
batching. The shared `(c, z)` pair is what collapses two accepting
transcripts to a single common `w` across all legs (special soundness:
`w = (z − z')/(c − c')` simultaneously for every leg). Identity bases are
rejected before proving or verifying — an identity base makes its leg
trivially satisfiable.

The prover MUST NOT use the RNG draw directly as `k`. It derives `k` under
`b"ve-capsule.contribute-dleq.nonce.v1"` from fresh 32-byte entropy, the
helper scalar `w`, the DLEQ sub-domain, the canonical `binding`, the ordered
bases, the ordered images, and a counter, then reduces the SHA-256 digest mod
`n` and rejects zero. This keeps honest proofs randomized while preventing a
repeated RNG byte stream, forked process, or embedded-RNG fault from reusing the
same Schnorr/Chaum-Pedersen nonce across two different challenges. Such reuse
would expose the helper scalar as `w = (z - z')/(c - c')`.

Item 3 pins the `Partial` to one capsule core (`C` and every `(E_j,D_j)`),
one composite key and gate list (`Y*`, `g*`), one gate, and one caller
context — it cannot be replayed elsewhere; the bases (item 4) additionally
bind the DLEQ statement to the capsule's exact segment masks. The `Partial`
wire decoder
(`Partial::from_canonical_bytes`) enforces the §1 canonicality rules
(`z < n` via `Scalar::from_repr`, strict point decode) before
verification, exactly as `Proof::from_canonical_bytes` does for the seal
proof. A decoded `Partial` is unauthenticated — the decoder asserts framing
and canonicality only; authenticity is `Partial::verify` against the capsule
core, gate, and context, which every opening path runs on each partial
before it is summed.

## 7. What this document does NOT cover

> **Scope.** Deliberate exclusions. The four codec rules below remain normative.

- The byte-exact wire encoding of each proof element — fixed in the
  implementation and pinned by tests (re-encode-equality on every decoder;
  the cross-device `Partial` format additionally by a known-answer vector,
  and the ratchet challenge derivation and generators by KAT). This document
  fixes the absorption order, the challenge rule, and the proof
  *equations*, which the wire layout must feed consistently. Four codec
  rules ARE normative here: (1) all proof bytes pass through **one decode
  door** (`Proof::from_canonical_bytes`) enforcing §1 canonicality; (2) the
  layout is **fixed-shape with zero length fields** — every count derives
  from the frozen §0/§4.1 constants, so no wire length is
  attacker-controlled; (3) no nested or self-describing sub-codec (no
  serde-style framed vectors inside the proof body); (4) trailing bytes and
  re-encode inequality are rejects.
- Threshold `Σ_i C_i == T` (multi-fragment) — out of v1 scope.
- The line-by-line as-built re-derivation of §4.4 against the shipped
  code — that is the as-built traceability artifact.

## References

- **[Ped91]** Pedersen, *Non-Interactive and Information-Theoretic Secure
  Verifiable Secret Sharing*, CRYPTO '91 — hiding/binding commitments.
- **[BP++]** Eagen, Kanjalkar, Ruffing, Nick, *Bulletproofs++: Next
  Generation Confidential Transactions via Reciprocal Set Membership
  Arguments*, EUROCRYPT 2024 ([eprint 2022/510](https://eprint.iacr.org/2022/510),
  rev. 2023-07-17) — the
  aggregated reciprocal range proof and weighted norm linear argument
  (§4.1). The 2023 revision's `δ` challenge fixes the 2022 draft's
  soundness hole; the revision history is part of why §2's schedule is
  regression-tested.
- **[CS-BPPP]** Cypher Stack (A. Feickert), *Bulletproofs++ review*, final
  report 2024-03-29, [github.com/cypherstack/bppp-review](https://github.com/cypherstack/bppp-review)
  — independent
  line-by-line review. Its corrected round-extraction lemma (the eprint's
  Lemma 5 is invalid as written) and corrected SHVZK proof are **normative
  for this implementation** alongside the paper. The review corrected those two
  items but declined to assert BP++'s overall soundness even as corrected (it
  also flagged Lemma 6 and the Theorem 1–2 witness-extended-emulation proofs as
  incorrect or incomplete); the construction inherits that residual, uncertified
  assumption.
- **[Mau09]** Maurer, *Unifying Zero-Knowledge Proofs of Knowledge*,
  AFRICACRYPT '09 — the generalized (multi-representation, shared-response)
  Schnorr Σ-protocol used for the linking sigma and carry residuals.
- **[CS03]** Camenisch, Shoup, *Practical Verifiable Encryption and
  Decryption of Discrete Logs*, CRYPTO '03 — the verifiable-encryption
  template.
- **[JUG]** Shlomovits, Leiba, *JugglingSwap: Scriptless Atomic Cross-Chain
  Swaps*, 2020 ([arXiv:2007.14423](https://arxiv.org/abs/2007.14423); ZenGo
  implementation
  [github.com/ZenGo-X/dlog-verifiable-enc](https://github.com/ZenGo-X/dlog-verifiable-enc))
  — segmented verifiable encryption of a secp256k1 scalar (8-bit segments,
  per-segment Bulletproofs + consistency sigma); the nearest predecessor of
  ec-segve's statement.
- **[Groth21]** Groth, *Non-Interactive Distributed Key Generation and Key
  Resharing*, [eprint 2021/339](https://eprint.iacr.org/2021/339) — chunked
  exponent-ElGamal recovered by BSGS with chunking range proofs (pairing
  setting, DFINITY NIDKG).
- **[TZ21]** Takahashi, Zaverucha, *Verifiable Encryption from
  MPC-in-the-Head*, [eprint 2021/1704](https://eprint.iacr.org/2021/1704) —
  transparent verifiable encryption of EC private keys, different proof
  technology.
- **[FKMV12]** Faust, Kohlweiss, Marson, Venturi, *On the Non-Malleability
  of the Fiat–Shamir Transform*, INDOCRYPT '12 — context for FS
  non-malleability. v1 does **not** invoke its quasi-unique-responses
  theorem (unique responses are not claimed for either proof system);
  cited to delimit what is NOT claimed (§4.4).
- **[PS00]** Pointcheval, Stern, *Security Arguments for Digital Signatures
  and Blind Signatures*, J. Cryptology '00 — the forking lemma (ROM
  knowledge-soundness of FS-transformed Σ-protocols, the §4.4 (K) basis).
- **[RFC9380]** Faz-Hernández et al., *Hashing to Elliptic Curves* — the
  derivation of `H`.
- DDH / ElGamal IND-CPA on a prime-order group — standard (e.g.
  Boneh–Shoup).
