poblar_capa_landing.py:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Landing (AVRO + Particionamiento)
Proyecto: HOUSING PRICES
"""

import sys
import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

# =============================================================================
# @section 1. Configuración de parámetros
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Landing (Housing)')
    parser.add_argument('--env', type=str, default='TopicosB', help='Entorno: DEV, QA, PROD')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base en HDFS')
    parser.add_argument('--schema_path', type=str, default='/user/hadoop/datalake/schema', help='Ruta de esquemas AVRO')
    parser.add_argument('--source_db', type=str, default='topicosb_workload', help='Base de datos origen (Workload)')
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession
# =============================================================================

def create_spark_session(app_name="ProcesoLanding_Housing-DiegoFlores"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("spark.sql.avro.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones auxiliares
# =============================================================================

def crear_database(spark, env, username, base_path):
    db_name = f"{env}_landing".lower()
    db_location = f"{base_path}/{username}/datalake/{db_name.upper()}"
    
    spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    print(f"✅ Database '{db_name}' creada en: {db_location}")
    return db_name

def crear_tabla_avro_hive(spark, db_name, table_name, location, schema_avsc_url, partitioned_by=None):
    partition_clause = ""
    if partitioned_by:
        partition_cols = ", ".join([f"{c} STRING" for c in partitioned_by])
        partition_clause = f"PARTITIONED BY ({partition_cols})"
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_name}
    {partition_clause}
    STORED AS AVRO
    LOCATION '{location}'
    TBLPROPERTIES (
        'avro.schema.url'='{schema_avsc_url}',
        'avro.output.codec'='snappy'
    )
    """
    spark.sql(create_sql)
    print(f"✅ Tabla AVRO '{db_name}.{table_name}' registrada con esquema: {schema_avsc_url}")

def insertar_datos_avro(spark, db_name, table_name, df_source, partition_col=None):
    # Estandarizar nombres a minúsculas
    df_source = df_source.toDF(*[c.lower() for c in df_source.columns])

    if partition_col:
        p_col = partition_col.lower()
        # Reordenar columnas: la columna de partición DEBE ir al final para Hive
        cols = [c for c in df_source.columns if c != p_col] + [p_col]
        df_to_insert = df_source.select(*cols)
        
        df_to_insert.createOrReplaceTempView("src_data")
        insert_sql = f"INSERT OVERWRITE TABLE {db_name}.{table_name} PARTITION ({p_col}) SELECT * FROM src_data"
    else:
        df_source.createOrReplaceTempView("src_data")
        insert_sql = f"INSERT OVERWRITE TABLE {db_name}.{table_name} SELECT * FROM src_data"

    spark.sql(insert_sql)

# =============================================================================
# @section 4. Configuración del Proyecto Housing
# =============================================================================

TABLAS_CONFIG = [
    {
        "nombre": "HOUSING",
        "archivo_avsc": "housing.avsc",
        "partitioned_by": ["ocean_proximity"], # Particionamos por ubicación
        "dynamic_partition": True
    }
]

# =============================================================================
# @section 5. Proceso principal
# =============================================================================

def procesar_tabla_landing(spark, args, db_landing, db_source, config):
    table_name = config["nombre"]
    print(f"📥 Procesando Landing: {table_name}")
    
    location = f"{args.base_path}/{args.username}/datalake/{db_landing.upper()}/{table_name.lower()}"
    # La ruta del AVSC debe coincidir con donde lo subiste en el instrucciones.txt
    schema_url = f"{args.schema_path}/{db_landing.upper()}/{config['archivo_avsc']}"
    
    # 1. Crear tabla AVRO en Hive
    crear_tabla_avro_hive(
        spark=spark,
        db_name=db_landing,
        table_name=table_name,
        location=location,
        schema_avsc_url=schema_url,
        partitioned_by=config["partitioned_by"]
    )
    
    # 2. Cargar desde Workload
    df_source = spark.table(f"{db_source}.{table_name}")
    
    # 3. Insertar con particionamiento
    insertar_datos_avro(
        spark=spark,
        db_name=db_landing,
        table_name=table_name,
        df_source=df_source,
        partition_col=config["partitioned_by"][0] if config["partitioned_by"] else None
    )
    
    print(f"✅ Finalizado: {table_name}")
    spark.sql(f"SELECT * FROM {db_landing}.{table_name} LIMIT 5").show()

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        db_landing = f"{args.env.lower()}_landing"
        crear_database(spark, args.env.lower(), args.username, args.base_path)
        
        for config in TABLAS_CONFIG:
            procesar_tabla_landing(spark, args, db_landing, args.source_db, config)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
