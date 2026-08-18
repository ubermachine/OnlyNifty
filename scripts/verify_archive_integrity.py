"""
Archive & Journal Hash-Chain Integrity Verifier.
Validates end-to-end cryptographic linkages and content reproducibility across all JSONL archives.
"""

import sys
import os
import glob

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.signal_journal import verify_archive_file

def main():
    archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "archive")
    if not os.path.exists(archive_dir):
        print(f"Archive directory not found: {archive_dir}")
        sys.exit(1)

    files = glob.glob(os.path.join(archive_dir, "*.jsonl"))
    if not files:
        print(f"No .jsonl archive files found in {archive_dir}")
        sys.exit(0)

    all_valid = True
    total_audited = 0

    print("=" * 65)
    print("  ONLYNIFTY CRYPTOGRAPHIC AUDIT VERIFIER")
    print("=" * 65)

    for filepath in sorted(files):
        fname = os.path.basename(filepath)
        res = verify_archive_file(filepath)
        total_audited += res["total_records"]
        
        status_str = "PASS [100% VALID]" if res["is_valid"] else "FAIL [TAMPER DETECTED]"
        print(f"\n[ARCHIVE] {fname}")
        print(f"  Records Audited     : {res['total_records']}")
        print(f"  Broken Links (prev) : {res['broken_links']}")
        print(f"  Content Mismatches  : {res['content_mismatches']}")
        print(f"  Audit Status        : {status_str}")

        if not res["is_valid"]:
            all_valid = False
            for err in res["errors"][:5]:
                print(f"    ERROR: {err}")
            if len(res["errors"]) > 5:
                print(f"    ... and {len(res['errors']) - 5} more errors")

    print("\n" + "=" * 65)
    if all_valid:
        print(f"SUCCESS: All {total_audited} records across {len(files)} archives verified cryptographically sound!")
        sys.exit(0)
    else:
        print("FAILED: Cryptographic audit check failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
