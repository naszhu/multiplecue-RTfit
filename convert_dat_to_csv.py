from pathlib import Path
import csv


# Set this to the folder (under exp/) that contains your .dat files.
SOURCE_SUBFOLDER = "data/ParticipantCPP002-001"

MIN_TAB_COLUMNS = 10
STRING_COLUMNS = {"Cues", "CueValues", "RespLoc"}
# If True, drop eye movement columns from "EyeT1" to the final column.
EXCLUDE_EYE_MOVEMENT_COLUMNS = True
# If True, overwrite existing CSV files instead of skipping them.
OVERWRITE_EXISTING_CSV = True


def find_table_start(lines: list[str], min_cols: int = MIN_TAB_COLUMNS) -> tuple[int, int]:
    """Return (header_idx, n_cols) for the first tabular section."""
    for i, line in enumerate(lines):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < min_cols:
            continue
        for j in range(i + 1, len(lines)):
            next_line = lines[j].strip()
            if not next_line:
                continue
            if len(lines[j].rstrip("\n").split("\t")) == len(parts):
                return i, len(parts)
            break
    raise ValueError("Could not find a tabular header line in .dat file.")


def convert_one(dat_path: Path, out_csv: Path) -> None:
    with dat_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_idx, n_cols = find_table_start(lines)

    rows = []
    for raw in lines[header_idx:]:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = raw.rstrip("\n").split("\t")
        if len(parts) == n_cols:
            rows.append(parts)

    if not rows:
        raise ValueError("No tabular rows found in .dat file.")

    header = rows[0]

    if EXCLUDE_EYE_MOVEMENT_COLUMNS and "EyeT1" in header:
        eye_start_idx = header.index("EyeT1")
        rows = [row[:eye_start_idx] for row in rows]
        header = rows[0]

    string_col_indices = [idx for idx, col in enumerate(header) if col in STRING_COLUMNS]

    # Prefix with apostrophe so downstream tools keep these as text (not numbers).
    for row in rows[1:]:
        for idx in string_col_indices:
            if idx >= len(row):
                continue
            value = row[idx].strip()
            if value and not value.startswith("'"):
                row[idx] = f"'{value}"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    source_dir = exp_dir / SOURCE_SUBFOLDER
    out_dir = source_dir / "extracted data"
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(source_dir.rglob("*.dat"))
    if not dat_files:
        print(f"No .dat files found in: {source_dir}")
        return

    converted = 0
    overwritten = 0
    skipped_existing = 0
    skipped_error = 0

    for dat_path in dat_files:
        out_csv = out_dir / f"{dat_path.stem}.csv"
        existed_before = out_csv.exists()
        if out_csv.exists() and not OVERWRITE_EXISTING_CSV:
            skipped_existing += 1
            continue
        try:
            convert_one(dat_path, out_csv)
            if existed_before and OVERWRITE_EXISTING_CSV:
                overwritten += 1
            else:
                converted += 1
        except Exception as e:
            skipped_error += 1
            print(f"Skip (error): {dat_path.name} -> {e}")

    print(
        f"Done. Converted new: {converted}, overwritten: {overwritten}, "
        f"skipped existing: {skipped_existing}, skipped errors: {skipped_error}"
    )
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
