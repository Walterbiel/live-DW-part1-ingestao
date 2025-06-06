# DW Airflow + DBT – Part One 🚀

Projeto‑laboratório usado na live **“Data Warehouse ao Vivo – Parte 1”**. Aqui você aprende a:

1. **Gerar dados sintéticos** de vendas e devoluções;  
2. Carregar a camada **bronze** em PostgreSQL;  
3. Disponibilizar um **endpoint FastAPI** que entrega lotes de vendas;  
4. **Orquestrar** ingestão de dados com **Apache Airflow**.

### Criar banco de dados postgres local ou pelo Render !

---

## Pré‑requisitos

| Ferramenta | Versão testada |
|------------|----------------|
| Python     | 3.11.x         |
| PostgreSQL | 15 ou ↑        |
| Apache Airflow | 2.8.4 |
| FastAPI    | 0.111 |
| Uvicorn    | 0.29 |
| Pandas     | 2.1 |
| NumPy      | 1.26 |

Instale as dependências:

```bash

# 1 ▸ criar e ativar o venv
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2 ▸ atualizar pip
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Passo a passo

# Scripts DDL – Camada **bronze**

> **Execute** estes comandos no PostgreSQL antes da ingestão.  
> Cada tabela recebe uma chave primária surrogate (`BIGSERIAL`) iniciando em 1.

---

## 1. Criar schema

```sql
CREATE SCHEMA IF NOT EXISTS bronze;
```

---

## 2. Tabela `bronze.vendas`

```sql
CREATE TABLE IF NOT EXISTS bronze.vendas (
    pk_vendas       BIGSERIAL PRIMARY KEY,
    id_venda        BIGINT        NOT NULL,
    id_produto      INTEGER       NOT NULL,
    preco           NUMERIC(12,2) NOT NULL,
    quantidade      INTEGER       NOT NULL,
    data_venda      DATE          NOT NULL,
    id_cliente      INTEGER,
    id_loja         INTEGER,
    id_vendedor     INTEGER,
    meio_pagamento  TEXT,
    parcelamento    SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_vendas_id_venda ON bronze.vendas (id_venda);
CREATE INDEX IF NOT EXISTS idx_vendas_data     ON bronze.vendas (data_venda);
CREATE INDEX IF NOT EXISTS idx_vendas_produto  ON bronze.vendas (id_produto);
```

---

## 3. Tabela `bronze.devolucoes`

```sql
CREATE TABLE IF NOT EXISTS bronze.devolucoes (
    pk_devolucao    BIGSERIAL PRIMARY KEY,
    id_venda        BIGINT    NOT NULL,
    id_produto      INTEGER   NOT NULL,
    preco           NUMERIC(12,2) NOT NULL,
    quantidade      INTEGER   NOT NULL,
    data_venda      DATE      NOT NULL,
    data_devolucao  DATE      NOT NULL,
    id_cliente      INTEGER,
    id_loja         INTEGER,
    id_vendedor     INTEGER,
    motivo          TEXT,
    UNIQUE (id_venda, id_produto)
);

CREATE INDEX IF NOT EXISTS idx_dev_data_devolucao
    ON bronze.devolucoes (data_devolucao);
```

---

## 4. Tabela `bronze.produtos`

```sql
CREATE TABLE IF NOT EXISTS bronze.produtos (
    pk_produto          BIGSERIAL PRIMARY KEY,
    id_produto          INTEGER UNIQUE NOT NULL,
    nome_produto        TEXT    NOT NULL,
    categoria           TEXT,
    percentual_imposto  NUMERIC(5,2)
);
```

---

## 5. Tabela `bronze.lojas`

```sql
CREATE TABLE IF NOT EXISTS bronze.lojas (
    pk_loja     BIGSERIAL PRIMARY KEY,
    id_loja     INTEGER UNIQUE NOT NULL,
    nome_loja   TEXT    NOT NULL,
    logradouro  TEXT,
    numero      INTEGER,
    bairro      TEXT,
    cidade      TEXT,
    estado      CHAR(2),
    cep         VARCHAR(10)
);
```

---

## 6. Tabela `bronze.vendedores`

```sql
CREATE TABLE IF NOT EXISTS bronze.vendedores (
    pk_vendedor     BIGSERIAL PRIMARY KEY,
    id_vendedor     INTEGER UNIQUE NOT NULL,
    nome_vendedor   TEXT    NOT NULL,
    data_admissao   DATE,
    endereco_vendedor TEXT,
    data_nascimento DATE
);
```
---


### 1. Gerar bases CSV

```bash
python gerador_base_venda.py          # gera base_vendas_2M.csv
python gerador_base_devolucoes.py     # gera base_devolucoes.csv
```

### 2. Carga inicial no PostgreSQL

```bash
python carga_inicial_postgres.py
```

A conexão usa o DSN definido no próprio script (`DB_URL`). Ajuste conforme seu banco.

### 3. API de vendas em lote

```bash

uvicorn api_vendas_batch:app --reload --port 8001
```

* **Endpoint:** `GET /vendas/{quantidade}`  
* **Exemplo:** `http://localhost:8001/vendas/500`

---
