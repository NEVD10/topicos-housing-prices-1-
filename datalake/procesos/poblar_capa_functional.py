#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Functional - Proyecto HOUSING
Generación de KPIs y métricas para análisis final (Capa Gold)
"""

import sys
import argparse
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, avg, round, max, min, count

# =============================================================================
# @section 1. Configuración de parámetros
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Functional (Housing)')
    parser.add_argument('--env', type=str, default='TopicosB', help='Entorno: DEV, QA, PROD')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user', help='Ruta base en HDFS')
    parser.add_argument('--source_db', type=str, default='curated', help='Base de datos origen')
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession
# =============================================================================

def create_spark_session(app_name="ProcesoFunctional_Housing-DiegoFlores"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones auxiliares
# =============================================================================

def crear_database(spark, env, username, base_path):
    db_name = f"{env}_functional".lower()
    db_location = f"{base_path}/{username}/datalake/{db_name.upper()}"
    spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    return db_name

# =============================================================================
# @section 4. Lógica de Enriquecimiento (KPIs de Vivienda)
# =============================================================================

def generar_metricas_vivienda(df):
    """
    Crea métricas derivadas útiles para el negocio:
    1. Precio por habitación
    2. Ratio de población por hogar
    3. Categorización por costo
    """
    logger.info("✨ Generando métricas derivadas (Feature Engineering)...")
    
    df_enriched = df.withColumn(
        "price_per_room", round(col("median_house_value") / col("total_rooms"), 2)
    ).withColumn(
        "pop_per_household", round(col("population") / col("households"), 2)
    ).withColumn(
        "is_high_value", when(col("median_house_value") > 400000, "HIGH").otherwise("STANDARD")
    )
    
    return df_enriched

def generar_kpi_ocean(df):
    """
    Genera un resumen estadístico por proximidad al océano
    """
    logger.info("📊 Calculando KPIs por proximidad al océano...")
    
    kpi_df = df.groupBy("ocean_proximity").agg(
        round(avg("median_house_value"), 2).alias("avg_price"),
        round(avg("median_income"), 2).alias("avg_income"),
        count("*").alias("total_properties"),
        round(max("median_house_value"), 2).alias("max_price")
    )
    
    return kpi_df

# =============================================================================
# @section 5. Ejecución principal
# =============================================================================

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        env_lower = args.env.lower()
        db_curated = f"{env_lower}_{args.source_db}"
        db_functional = f"{env_lower}_functional"
        
        # 1. Crear Base de Datos
        crear_database(spark, env_lower, args.username, args.base_path)
        
        # 2. Leer data de la capa Curated
        df_housing = spark.table(f"{db_curated}.housing")
        
        # 3. Paso de Enriquecimiento (Tabla Gold de Viviendas)
        df_final = generar_metricas_vivienda(df_housing)
        
        # Guardar Tabla Enriquecida
        df_final.write.mode("overwrite") \
            .format("parquet") \
            .saveAsTable(f"{db_functional}.housing_enriched")
        
        # 4. Paso de Agregación (Tabla de KPIs para el Informe)
        df_kpi = generar_kpi_ocean(df_final)
        
        df_kpi.write.mode("overwrite") \
            .format("parquet") \
            .saveAsTable(f"{db_functional}.kpi_ocean_proximity")
            
        logger.info("✅ Capa Functional completada exitosamente.")
        
        # Mostrar resultados para los Anexos del informe
        print("\n--- MUESTRA TABLA ENRIQUECIDA (GOLD) ---")
        spark.sql(f"SELECT * FROM {db_functional}.housing_enriched LIMIT 5").show()
        
        print("\n--- KPI: PRECIO PROMEDIO POR CERCANÍA AL MAR ---")
        df_kpi.show()

    except Exception as e:
        logger.error(f"❌ Error en Functional: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()