import psycopg2
import pandas as pd
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

# Leitura dos arquivos
df_lojas = pd.read_excel("BAses para live DW.xlsx", sheet_name="lojas")
df_produtos = pd.read_excel("BAses para live DW.xlsx", sheet_name="produtos")
df_vendedores = pd.read_excel("BAses para live DW.xlsx", sheet_name="vendedores")
df_vendas = pd.read_csv("base_vendas_2M.csv")
df_devolucoes = pd.read_csv("base_devolucoes.csv")

# Função para truncar e inserir dados
def inserir_dados(df, tabela, colunas):
    try:
        tqdm.write(f"🔁 Limpando bronze.{tabela}...")
        cur.execute(f"TRUNCATE TABLE bronze.{tabela}")
        tqdm.write(f"✅ Tabela bronze.{tabela} truncada com sucesso.")

        values = [tuple(row[col] for col in colunas) for _, row in df.iterrows()]
        placeholders = ','.join(['%s'] * len(colunas))
        sql = f"INSERT INTO bronze.{tabela} ({', '.join(colunas)}) VALUES ({placeholders})"
        tqdm.write(f"📥 Inserindo bronze.{tabela}...")
        cur.executemany(sql, values)
        tqdm.write(f"✅ Inserido: bronze.{tabela}")
    except Exception as e:
        tqdm.write(f"❌ Erro ao inserir bronze.{tabela}: {e}")
        conn.rollback()

# Inserções 
inserir_dados(df_lojas, "lojas", [
    "id_loja", "nome_loja", "logradouro", "numero", "bairro", "cidade", "estado", "cep"
])

inserir_dados(df_produtos, "produtos", [
    "id_produto", "nome_produto", "categoria", "percentual_imposto"
])

inserir_dados(df_vendedores, "vendedores", [
    "id_vendedor", "nome_vendedor", "data_admissao", "endereco_vendedor", "data_nascimento"
])

inserir_dados(df_vendas, "vendas", [
    "id_venda", "id_produto", "preco", "quantidade", "data_venda",
    "id_cliente", "id_loja", "id_vendedor", "meio_pagamento", "parcelamento"
])

inserir_dados(df_devolucoes, "devolucoes", [
    "id_devolucao", "id_venda", "data_devolucao", "motivo", "quantidade", "valor_total"
])

# Encerrar conexão
cur.close()
conn.close()
print("🏁 Carga de todas as tabelas finalizada.")
