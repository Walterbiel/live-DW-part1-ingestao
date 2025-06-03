import pandas as pd

# Arquivos
excel_path = "BAses para live DW.xlsx"
csv_vendas = "base_vendas_2M.csv"
csv_devolucoes = "base_devolucoes.csv"

# Dados
df_lojas = pd.read_excel(excel_path, sheet_name="lojas")
df_produtos = pd.read_excel(excel_path, sheet_name="produtos")
df_vendedores = pd.read_excel(excel_path, sheet_name="vendedores")
df_vendas = pd.read_csv(csv_vendas)
df_devolucoes = pd.read_csv(csv_devolucoes)

# URI do PostgreSQL (string)
pg_uri = "postgresql+psycopg2://postgresadmin:jAloy0oZD2Aks1zMjCgDDilQExPLXKCg@dpg-d0khna8gjchc73ae1r1g-a.oregon-postgres.render.com:5432/general_rtxt"

# Parâmetros
common = {
    "con": pg_uri,
    "if_exists": "replace",
    "index": False,
    "method": "multi",
    "chunksize": 100_000
}

# Carga
df_lojas.to_sql("lojas", **common)
df_produtos.to_sql("produtos", **common)
df_vendedores.to_sql("vendedores", **common)
df_vendas.to_sql("vendas", **common)
df_devolucoes.to_sql("devolucoes", **common)

print("✅ Carga concluída com sucesso usando URI string e pandas.")
