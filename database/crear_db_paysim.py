from pathlib import Path
import argparse
import sqlite3
import time

import pandas as pd

REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def create_database(csv_path, db_path, chunk_size=100000, replace=False):
    csv_path = Path(csv_path).resolve()
    db_path = Path(db_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro: {csv_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not replace:
        raise FileExistsError(
            f"Ya existe {db_path}. Usa --reemplazar para crearla de nuevo."
        )
    if db_path.exists():
        db_path.unlink()

    start = time.perf_counter()
    total = 0

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")

        for block_number, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunk_size), start=1
        ):
            missing = sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))
            if missing:
                raise ValueError(f"Faltan columnas: {missing}")

            chunk = chunk[REQUIRED_COLUMNS].copy()
            chunk.insert(0, "id", range(total + 1, total + len(chunk) + 1))
            chunk.to_sql(
                "transactions",
                connection,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=2000,
            )
            total += len(chunk)
            print(f"Bloque {block_number}: {total:,} filas")

        print("Creando indices...")
        connection.execute("CREATE UNIQUE INDEX idx_id ON transactions(id)")
        connection.execute("CREATE INDEX idx_step ON transactions(step)")
        connection.execute("CREATE INDEX idx_fraud ON transactions(isFraud)")
        connection.execute("CREATE INDEX idx_type ON transactions(type)")
        connection.execute("CREATE INDEX idx_origin ON transactions(nameOrig)")
        connection.execute("CREATE INDEX idx_destination ON transactions(nameDest)")
        connection.commit()

        rows, frauds = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(isFraud), 0) FROM transactions"
        ).fetchone()

    elapsed = time.perf_counter() - start
    print(f"Base creada: {db_path}")
    print(f"Filas: {rows:,}")
    print(f"Fraudes: {frauds:,}")
    print(f"Tiempo: {elapsed:,.1f} segundos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte PaySim CSV a SQLite")
    parser.add_argument("--csv", default="paysim.csv")
    parser.add_argument("--db", default="database/paysim.db")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--reemplazar", action="store_true")
    args = parser.parse_args()
    create_database(args.csv, args.db, args.chunksize, args.reemplazar)
