import psycopg2
import pandas as pd
from io import StringIO
from tqdm import tqdm

# Conexão com o PostgreSQL
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

try:
    tqdm.write("🔁 TRUNCATE bronze.vendas...")
    cur.execute('TRUNCATE TABLE bronze.vendas')
    tqdm.write("✅ Tabela limpa com sucesso.")

    tqdm.write("📥 Lendo CSV...")
    df_vendas = pd.read_csv("base_vendas_2M.csv")

    colunas = [
        "id_venda", "id_produto", "preco", "quantidade", "data_venda",
        "id_cliente", "id_loja", "id_vendedor", "meio_pagamento", "parcelamento"
    ]
    df_vendas = df_vendas[colunas]

    tqdm.write("📦 Convertendo dataframe para CSV em memória...")
    buffer = StringIO()
    df_vendas.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    tqdm.write("🚀 Executando COPY para bronze.vendas...")
    cur.copy_expert(f"""
        COPY bronze.vendas ({', '.join(colunas)})
        FROM STDIN WITH (FORMAT CSV)
    """, buffer)

    tqdm.write("✅ Inserção concluída com sucesso.")

except Exception as e:
    tqdm.write(f"❌ Erro durante inserção: {e}")
    conn.rollback()

finally:
    cur.close()
    conn.close()
    tqdm.write("🏁 Processo finalizado.")
