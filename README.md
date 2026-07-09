# ve-capsule

> [!WARNING]
> **Research toy.** ve-capsule is experimental and unaudited — not reviewed, not
> stable, and not safe for protecting real keys yet. Explore freely; don't rely on it.

Verifiable encryption on **secp256k1** — seal a secret so that *anyone* can check
it matches a public commitment, without opening it.

When several parties or devices each encrypt a piece of a recovery share,
ciphertext alone is not a security model: you are hoping the others sealed the
right thing. ve-capsule attaches a public proof to each seal. Peers can audit a
capsule at setup — recipient offline — and lying (or honest mistakes) fails in
front of everyone who has an interest in the outcome. Catching your own backup
mistakes is a free side effect; the multi-party guarantee is the point.

Everything stays on the native curve Bitcoin, FROST, and Taproot already speak.
No class group, pairing curve, or RSA modulus. Gated recovery can be authorized
with an ordinary secp256k1 signature. The built-in verifier is BIP-340 Schnorr;
FROST(secp256k1)-TR signatures verify as BIP-340 under the caller-supplied
x-only key (Taproot key-path users supply the tweaked output key).

| | |
|---|---|
| **Curve** | secp256k1 only |
| **Capsule size** | 5,414 bytes fixed (frozen params) |
| **Strip core** | 778 bytes (core envelope + ElGamal limbs) |
| **Recovery hint** | 65 bytes per piece |
| **Unsafe** | `#![forbid(unsafe_code)]` |
| **Spec** | [`docs/ec-segve-soundness.md`](docs/ec-segve-soundness.md) |

Site write-up: [jud.me/ve-capsule](https://www.jud.me/ve-capsule/).

---

## What problem this solves

**Encryption** gives you opaque bytes and a promise between sealer and recipient.

**Verifiable encryption** adds a public check: given the ciphertext and a target
public key `M = m·G`, anyone can confirm the encrypted limbs reassemble to the
scalar behind `M` — without opening the ciphertext, and without the recipient’s
secret. Secrecy is relative to that public commitment: the proof and ciphertext
leak nothing beyond the already-public discrete-log instance `M`.

That matters when:

- several participants each seal a share, and each needs the others to be honest
  (or at least correct) *before* anyone walks away from setup;
- the recovery device is offline during setup, but the group still wants an audit;
- recovery should optionally require consent (extra keys that must contribute
  before open);
- after audit you want to drop multi-kilobyte proofs and keep a small cold-storage
  payload.

---

## Terms

| Word | Meaning here |
|---|---|
| **Scalar** | A secret number on the curve — a private key or key share (32 bytes). |
| **Public key** | `scalar · G` — the public partner of that secret. |
| **Capsule** | One sealed scalar: ciphertext + proof + framing. ~5.4 KB of canonical bytes. |
| **Case** | Several capsules whose secrets *add up* to one target key. One group check: `Σ Mⱼ == M`. |
| **Recipient** | The public key that can eventually open the capsule. |
| **Gate / authorizer** | An extra public key that must contribute before open (optional). |
| **Context (`ctx`)** | Application binding (package id, epoch, purpose…) so capsules cannot be replayed elsewhere. |
| **Verify** | Check the proof + claim. No secrets; no plaintext beyond public `M = m·G`. |
| **Unseal / open** | Decrypt after verify (and after any gate contributions). |
| **Strip** | Drop the proof; keep the opening core (~778 B). |
| **Recovery hint** | 65-byte ECDH shadow of one piece — recipient-only recovery for minimal openers. |

Flow:

```
seal  →  (hand around bytes)  →  verify (anyone)  →  unseal (recipient [+ gates])
```

Proof algebra is crate-private. Callers only see seal / verify / contribute /
unseal. Secrets zeroize on drop.

---

## Quick start

```rust
use ve_capsule::{Capsule, PrivateKey};

// Application-defined Context binds the capsule (package id, epoch, …).
// let ctx = …;

let secret    = PrivateKey::from_secret(/* 32 bytes */)?;
let recipient = PrivateKey::from_secret(/* 32 bytes */)?;
let pk        = secret.public_key();
let y         = recipient.public_key();

// Seal
let capsule = Capsule::builder(&secret, &y, &ctx).seal()?;
let bytes   = capsule.to_canonical_bytes(); // store / transport these

// Anyone can audit — no secrets
let capsule = Capsule::from_canonical_bytes(&bytes)?;
capsule.verify_ungated(&pk, &y, &ctx)?;
// Proven: limbs reassemble to the dlog of `pk` (= M). Still sealed.

// Recipient opens
let verified = capsule.verify_ungated(&pk, &y, &ctx)?;
let opened   = verified.unseal(&recipient, &[])?;
assert_eq!(opened.public_key(), pk);
```

---

## Multi-party: a Case of piece-capsules

When a secret is already additive pieces `s = Σ sⱼ` (threshold / MPC recovery),
each party seals only its piece. Bundle the capsules into a **Case**. Every piece
shares the same `(recipient, access policy, ctx)` and differs only in its
commitment `Mⱼ = sⱼ·G`.

```rust
let case     = Case::new(capsules)?;
let verified = case.verify(&target_public, &recipient, &[], &ctx)?;
// Every piece present; Σ Mⱼ == target_public.
```

One missing or wrong piece fails the sum. That is the group audit: peers check
the Case at setup without opening any piece.

---

## Consent gates

With no gates, the recipient opens alone. With gates, every named access key
must contribute a `Partial` first. A gate can itself be a multi-party key — the
capsule only sees one access public key; the group produces its contribution
together via `contribute_for_gate`.

```rust
let capsule = Capsule::builder(&secret, &recipient, &ctx)
    .access_key(&authorizer) // or .access_keys(&[…])
    .seal()?;

let verified = capsule.verify(&public_key, &recipient, &[authorizer], &ctx)?;
let partial  = verified.contribute(&authorizer_key)?;
let opened   = verified.unseal(&recipient_secret, &[partial])?;
```

`Partial`s are independent, canonically serializable artifacts — gather them
across systems. `unseal` verifies each partial’s DLEQ against the verified
capsule before summing. The DLEQ itself is public (no secret leakage beyond
what the contribution already is); the usual integration path is contribute →
unseal, not a separate public-API verify step.

### Why stripped cores need a signature

After `strip`, the multi-kilobyte proof is gone. Without an attestation, an
attacker could hand an authorizer a fabricated core and harvest a partial
decryption (a **static-DH oracle** on the authorizer’s key). For gated recovery
of a stripped core, `verify_signed` checks a quorum **signature** over the
canonical attestation statement: core digest, recipient, gate roster commitment,
composite key, context, and params. BIP-340 Schnorr is the built-in scheme; a
FROST(secp256k1)-TR quorum signature verifies as BIP-340 under the x-only key
the caller supplies. Recipient-only recovery needs no signature: nobody
contributes, so there is no oracle to open.

---

## Compact recovery (drop the proofs)

The proof is for auditors at setup. Recovery does not need it. Two compact
forms remain, and they are not interchangeable.

**Strip — the default.** Keep a 778-byte opening core per capsule (core
envelope + ElGamal limbs): the same ciphertexts the group verified, nothing
new to trust. Re-anchors on the certified commitment when opened, and it is
the only compact form that supports gates.

**Recovery hints — for minimal openers.** Opening a core still means eleven
ElGamal decryptions and a bounded-dlog search; a hint opens with one ECDH, one
hash, and one subtraction. That puts the recovery arithmetic within reach of an
opener that is barely a computer — a secure element, a single-purpose recovery
device. Beside each capsule, seal a **65-byte** hint (one ECDH point + one
scalar); hints assemble into a recipient-only payload, and the ~5.4 KB proofs
stay behind. The proof does not cover hints: a bad hint can never yield a wrong
secret (recovery fails closed), but it can fail to yield one — keep the cores
if you need the audited fallback, and keep gated recovery on the strip path.

```rust
let rctx = RecoveryContext {
    certified_target: &target_public,
    ctx: &context_bytes,
    epoch,
};

let h1 = seal_recovery_hint(&contribution1, &recipient, &rctx, 1, &mut rng)?;
let h2 = seal_recovery_hint(&contribution2, &recipient, &rctx, 2, &mut rng)?;
let payload_bytes =
    assemble_recipient_recovery_payload(&target_public, &[(1, h1), (2, h2)])?;

// …later…
let payload   = CompactRecoveryPayload::from_canonical_bytes(&payload_bytes)?;
let recovered = recover_recipient_secret_from_payload(&payload, &recipient_secret, &rctx)?;
assert_eq!(recovered.public_key(), target_public); // or fail closed
```

This path is **self-securing**: recovery checks `s·G` against the **certified**
target, never a target taken on trust from the payload. Tampering yields a wrong
scalar that fails closed — not a wrong secret that looks right. Apply
interpolation weights *before* sealing if your share representation needs them.

---

## What’s in a capsule (wire layout)

Frozen params: `ℓ = 24` limb bits, `L = 11` limbs (`Params::FROZEN`). Every point
is 33-byte SEC1; every scalar is 32-byte big-endian. Fixed layout — no length
fields. Pin checked in tests: proof body **5,363** bytes + envelope **51** →
**5,414** bytes total.

| Component | Bytes | Role |
|---|---:|---|
| Envelope | 51 | Magic `ve-capsule.cap.v1`, version, commitment `C = m·G` |
| Segmented EC-ElGamal | 726 | 11 limb ciphertexts `(Eₖ, Dₖ)` — what recovery opens |
| Pedersen limb commitments | 726 | Value + complement per limb (range statement inputs) |
| Bulletproofs++ | 688 | One aggregated reciprocal range proof (log-size) |
| Carry chain | 1,045 | Exact integer decomposition `m + m̄ = n − 1` |
| Linking sigma | 2,178 | Binds ElGamal, Pedersen commitments, and `C` under one FS challenge |

After verify: strip → **778 B** core, or **65 B** hint per piece. Proof-stripped
recovery needs the ciphertexts; hint recovery needs the separately sealed
shadows. Both re-anchor on the certified target. The linking sigma is most of
the blob for the same reason the proof stack is most of the crate: binding
ciphertext to claim is the product.

---

## Construction

ve-capsule is **segmented EC-ElGamal verifiable encryption** (ec-segve-v1) on
secp256k1, in the Camenisch–Shoup VE tradition.

1. **Segment.** Split the scalar into `L` limbs of `ℓ` bits; encrypt each under
   curve ElGamal to the recipient (or composite gated key).
2. **Range.** One aggregated **Bulletproofs++** reciprocal range proof covers
   every limb and carry — logarithmic in the statement, not linear. Exact range
   = decryption search window (zero slack): a verifying capsule is a decryptable
   one.
3. **Carry.** Prove the limb decomposition is exact integer arithmetic.
4. **Link.** A linking sigma binds limbs, Pedersen commitments, and `C` under
   one Fiat–Shamir challenge so the range proof is about the same scalar the
   ciphertext encrypts.
5. **Gates.** Coefficiented multi-key seal; each authorizer `Partial` carries
   a batched DLEQ on its **own** FS transcript (§6 of the soundness spec) —
   not the seal proof’s multi-squeeze.
6. **Seal transcript.** Strict absorb-everything FS with byte-pinned
   challenges. NUMS generators re-derived locally; no trusted setup.

BP++ is the reciprocal argument of
[Eagen, Kanjalkar, Ruffing, and Nick](https://eprint.iacr.org/2022/510)
(EUROCRYPT 2024), with corrections from
[Cypher Stack’s review](https://github.com/cypherstack/bppp-review). That review
did not positively certify BP++’s knowledge-soundness; ve-capsule adopts the
corrected statements and inherits that residual, independently-uncertified risk.

The nearest prior instantiation is ZenGo’s
[Juggling](https://arxiv.org/abs/2007.14423) (Shlomovits and Leiba, 2020;
[implementation](https://github.com/ZenGo-X/dlog-verifiable-enc)): the same
statement on secp256k1 — segment the scalar, encrypt limbs under EC-ElGamal,
range-prove each, sigma-bind to the dlog of a public point. ve-capsule’s deltas
are one aggregated Bulletproofs++ argument across all limbs and carries in place
of per-segment range proofs, an exact-integer carry chain making the reassembled
scalar canonical (`m < n` over the integers, not just mod `n`), and the product
layer around it — consent gates, proof-stripping with quorum attestation,
recovery hints, a frozen wire format. Adjacent: Groth’s chunked exponent-ElGamal
recovered by BSGS ([eprint 2021/339](https://eprint.iacr.org/2021/339), pairing
setting, DFINITY NIDKG) and verifiable encryption of EC keys from MPC-in-the-head
(Takahashi–Zaverucha, [eprint 2021/1704](https://eprint.iacr.org/2021/1704),
different proof technology).

Rough laptop order of magnitude: verify ~150 ms; open much faster (BSGS over
limb windows). Seal where the plaintext already lives; verify anywhere public
data is available. The claimed properties are knowledge-soundness and
zero-knowledge in the random-oracle model; simulation-sound extractability and
UC composition are not claimed.

Full argument: [`docs/ec-segve-soundness.md`](docs/ec-segve-soundness.md).

---

## Security notes (integrations must read)

**Untrusted keys.** Recipient and access keys arrive as public points. The crate
aggregates gated keys under deterministic coefficients and rejects publicly
enumerable / bounded algebraic relations between components — a rejection the
spec scopes to its admitted capped-roster profile, not larger rosters. That is
not a substitute for enrollment: possession-certify and enrollment-bind keys;
require independent generation (no party knows a relation among secrets) before
treating keys as authorization material.

**Side channels.** Not constant-time. Seal and unseal branch on secret data
(digit decomposition; bounded dlog recovery). A fine-grained local observer
(timing, cache, power) may learn the scalar. Run seal where the plaintext
already is; unseal on the recipient; verify on public data only.

**What the proof gives you.** Knowledge-sound binding of ciphertext to the
claimed public key, recipient, exact gate roster, and context. It does not
replace higher-level policy (who is allowed to be a recipient, when recovery is
authorized, how keys are provisioned) and does not hide guessable secrets beyond
the public commitment `M = m·G`.

---

## Public surface

| Type | Role |
|---|---|
| `Capsule` / `CapsuleBuilder` | Sealed ciphertext + proof; builder for seal. |
| `VerifiedCapsule` | Post-verify capability: `contribute` / `unseal` / `strip`. |
| `Case` / `VerifiedCase` | Additive piece-capsules; sum check `Σ Mⱼ == M`. |
| `StrippedCapsule` / `StrippedCase` | Proof dropped — compact opening core. |
| `BoundCapsule` / `BoundCase` | Core re-pinned to target; `unseal` or `verify_signed`. |
| `CompactRecoveryPayload` | Assembled recipient-only hints. |
| `RecoveryHint` / `RecoveryContext` | 65-byte piece shadow + certified target/ctx/epoch. |
| `Partial` | One authorizer contribution; canonically serializable. |
| `Signature` / `Backing` / `Scheme` | Quorum attestation for stripped gated cores (BIP-340 today). |
| `PrivateKey` / `PublicKey` | Zeroizing scalar; validated point. |
| `Context` | App transcript binding. |
| `Params` | Frozen `(ℓ, L, d, D)` — `Params::FROZEN`. |
| `Error` | `PointDecode`, `DegenerateInput`, `Verification` (`#[non_exhaustive]`). |
| `Seal` / `Contribute` / `Unseal` | Verb traits mirroring inherent methods. |

---

## Install

Not on crates.io yet:

```toml
[dependencies]
ve-capsule = { git = "https://github.com/Jud/ve-capsule" }
```

## License

Licensed under either of Apache-2.0 or MIT, at your option.
