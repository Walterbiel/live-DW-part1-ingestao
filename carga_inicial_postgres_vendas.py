import psycopg2
import pandas as pd
from tqdm import tqdm

# Conexão
conn = psycopg2.connect(
    dbname="general_rtxt",
    user="postgresadmin",
    password="jAloy0oZD2Aks1zMjCgDDilQExPLXKCg",
    host="dpg-d0khna8gjchc73ae1r1g-a.oregon-postgres.render.com",
    port="5432",
    sslmode="require"
)
conn.autocommit = True
cur = conn.cursor()

# TRUNCATE
print("🔁 Limpando bronze.vendas...")
cur.execute("TRUNCATE TABLE bronze.vendas")
print("✅ Tabela bronze.vendas truncada.")

# Leitura em chunks
chunk_size = 100_000
colunas = [
    "id_venda", "id_produto", "preco", "quantidade", "data_venda",
    "id_cliente", "id_loja", "id_vendedor", "meio_pagamento", "parcelamento"
]
placeholders = ','.join(['%s'] * len(colunas))
sql = f"INSERT INTO bronze.vendas ({', '.join(colunas)}) VALUES ({placeholders})"

# Inserir em chunks
try:
    for i, chunk in enumerate(pd.read_csv("base_vendas_2M.csv", chunksize=chunk_size)):
        values = [tuple(row[col] for col in colunas) for _, row in chunk.iterrows()]
        tqdm.write(f"📦 Inserindo chunk {i+1}...")
        cur.executemany(sql, values)
        tqdm.write(f"✅ Chunk {i+1} inserido.")
except Exception as e:
    tqdm.write(f"❌ Erro ao inserir vendas: {e}")
    conn.rollback()

# Encerrar conexão
cur.close()
conn.close()
print("🏁 Carga da fato vendas finalizada.")
