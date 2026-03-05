#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Curated - Proyecto HOUSING
Transformación de tipos y limpieza de datos (Parquet + Snappy)
"""

import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, when, lit, avg

# =============================================================================
# @section 1. Configuración de parámetros
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Curated (Housing)')
    parser.add_argument('--env', type=str, default='TopicosB', help='Entorno: DEV, QA, PROD')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base en HDFS')
    parser.add_argument('--source_db', type=str, default='landing', help='Base de datos origen')
    parser.add_argument('--enable-validation', action='store_true', default=True, help='Activar validaciones')
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession
# =============================================================================

def create_spark_session(app_name="ProcesoCurated_Housing-DiegoFlores"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones de Calidad y Transformación
# =============================================================================

def crear_database(spark, env, username, base_path):
    db_name = f"{env}_curated".lower()
    db_location = f"{base_path}/{username}/datalake/{db_name.upper()}"
    spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    return db_name

def crear_tabla_parquet_hive(spark, db_name, table_name, schema_spark, location, partitioned_by=None):
    columns_def = [f"{field.name} {field.dataType.simpleString().upper()}" for field in schema_spark.fields]
    partition_clause = f"PARTITIONED BY ({', '.join([f'{c} STRING' for c in partitioned_by])})" if partitioned_by else ""
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_name} (
        {', '.join(columns_def)}
    )
    {partition_clause}
    STORED AS PARQUET
    LOCATION '{location}'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    spark.sql(create_sql)

def aplicar_reglas_calidad_housing(df, enable_validation=True):
    """
    Transformaciones para HOUSING:
    1. Casteo de String a Double/Integer
    2. Limpieza de nulos en total_bedrooms (se llena con el promedio)
    3. Filtro de valores coherentes
    """
    # 1. Casteo y selección
    df_transformed = df.select(
        col("longitude").cast(DoubleType()),
        col("latitude").cast(DoubleType()),
        col("housing_median_age").cast(DoubleType()),
        col("total_rooms").cast(DoubleType()),
        col("total_bedrooms").cast(DoubleType()),
        col("population").cast(DoubleType()),
        col("households").cast(DoubleType()),
        col("median_income").cast(DoubleType()),
        col("median_house_value").cast(DoubleType()),
        col("ocean_proximity").cast(StringType()) # Columna de partición
    )
    
    # 2. Manejo de Nulos en total_bedrooms (Punto crítico del dataset)
    mean_bedrooms = df_transformed.select(avg("total_bedrooms")).first()[0]
    df_transformed = df_transformed.fillna({"total_bedrooms": mean_bedrooms})
    
    if enable_validation:
        # 3. Filtros de Calidad: Precios positivos y coordenadas válidas
        df_transformed = df_transformed.filter(
            (col("median_house_value") > 0) & 
            (col("latitude").between(32, 42)) & # Rango de California
            (col("longitude").between(-125, -114))
        )
    
    return df_transformed

# =============================================================================
# @section 4. Esquemas y Configuración
# =============================================================================

SCHEMAS_CURATED = {
    "HOUSING": StructType([
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("housing_median_age", DoubleType(), True),
        StructField("total_rooms", DoubleType(), True),
        StructField("total_bedrooms", DoubleType(), True),
        StructField("population", DoubleType(), True),
        StructField("households", DoubleType(), True),
        StructField("median_income", DoubleType(), True),
        StructField("median_house_value", DoubleType(), True)
    ])
}

TABLAS_CONFIG = [
    {
        "nombre": "HOUSING",
        "partitioned_by": ["ocean_proximity"],
        "func_calidad": aplicar_reglas_calidad_housing
    }
]

# =============================================================================
# @section 5. Proceso Principal
# =============================================================================

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        db_curated = f"{args.env.lower()}_curated"
        db_source = f"{args.env.lower()}_{args.source_db}"
        
        crear_database(spark, args.env.lower(), args.username, args.base_path)
        
        for config in TABLAS_CONFIG:
            table_name = config["nombre"]
            location = f"{args.base_path}/{args.username}/datalake/{db_curated.upper()}/{table_name}"
            
            # Crear Estructura
            crear_tabla_parquet_hive(spark, db_curated, table_name, SCHEMAS_CURATED[table_name], location, config["partitioned_by"])
            
            # Cargar y Transformar
            df_source = spark.table(f"{db_source}.{table_name}")
            df_curated = config["func_calidad"](df_source, args.enable_validation)
            
            # Guardar en Parquet (asegurando que la partición vaya al final)
            p_col = config["partitioned_by"][0]
            cols = [c for c in df_curated.columns if c != p_col] + [p_col]
            
            df_curated.select(*cols).write.mode("overwrite").insertInto(f"{db_curated}.{table_name}")
            
            print(f"✅ Tabla {table_name} procesada en Curated.")
            spark.sql(f"SELECT * FROM {db_curated}.{table_name} LIMIT 5").show()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()