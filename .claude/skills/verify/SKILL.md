---
name: verify
description: Run the full Tempura verification gate — tests, docs, clippy. Use after any code change to confirm nothing is broken.
---

Run the complete verification pipeline in this exact order. Stop at the first failure.

## Step 1: Compile check

```bash
cargo check 2>&1
```

If this fails, fix compilation errors before proceeding.

## Step 2: Test suite

```bash
cargo test 2>&1
```

All 113+ tests must pass. If any test fails, diagnose the root cause — do NOT weaken the test.

## Step 3: Strict rustdoc

```bash
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps 2>&1
```

Every public item must have `///` doc comments. Zero warnings allowed.

## Step 4: Clippy

```bash
cargo clippy -- -D warnings 2>&1
```

Zero clippy warnings allowed.

## Report

After all steps pass, report a one-line summary:

> ✅ Verification passed: N tests, clean docs, clean clippy.

If any step failed, report which step failed and the specific error, then offer to fix it.
