import pandas as pd
import pdfplumber
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FoodRecord:
    """Estructura de datos para un registro de alimento."""
    codigo: str
    nombre: str
    valores: List[str]

class OptimizedNutritionExtractor:
    def __init__(self):
        self.columns = [
            'CODIGO', 'NOMBRE_ALIMENTO', 'ENERGIA_KCAL', 'ENERGIA_KJ', 'AGUA_G',
            'PROTEINAS_G', 'GRASA_TOTAL_G', 'CARBOHIDRATOS_TOTALES_G',
            'CARBOHIDRATOS_DISPONIBLES_G', 'FIBRA_DIETARIA_G', 'CENIZAS_G',
            'CALCIO_MG', 'FOSFORO_MG', 'ZINC_MG', 'HIERRO_MG', 'BETA_CAROTENO_UG',
            'VITAMINA_A_UG', 'TIAMINA_MG', 'RIBOFLAVINA_MG', 'NIACINA_MG',
            'VITAMINA_C_MG', 'ACIDO_FOLICO_UG', 'SODIO_MG', 'POTASIO_MG'
        ]

        # Compilar regex una sola vez
        self.CODE_START_PATTERN = re.compile(r'^([A-Z])\s+(\d+)\s+')
        self.CODE_END_PATTERN = re.compile(r'\s([A-Z])\s+(\d+)$')
        self.NUMBER_PATTERN = re.compile(r'^[\d.,]+$')
        self.INVALID_NUMBER_PATTERN = re.compile(r'^[.,]+$')

        # Set de valores faltantes para búsqueda O(1)
        self.MISSING_VALUES: Set[str] = {'.', '•', '-', '—', 'TR', 'tr', '<'}

        # Cache para códigos procesados
        self._code_cache: Dict[str, str] = {}

    @lru_cache(maxsize=1000)
    def _format_code(self, letter: str, number: str) -> str:
        """Formatea un código con cache."""
        return f"{letter} {number}"

    def _extract_code(self, line: str, position: str = 'start') -> Optional[str]:
        """Extrae código optimizado con regex precompilados."""
        pattern = self.CODE_START_PATTERN if position == 'start' else self.CODE_END_PATTERN
        match = pattern.search(line.strip())
        return self._format_code(match.group(1), match.group(2)) if match else None

    def _is_valid_number(self, token: str) -> bool:
        """Verifica si un token es un número válido."""
        return bool(self.NUMBER_PATTERN.match(token) and not self.INVALID_NUMBER_PATTERN.match(token))

    def _process_line_tokens(self, line: str, skip_code: bool = False) -> Tuple[str, List[str]]:
        """Procesa tokens de una línea separando nombre y valores."""
        if skip_code:
            line = self.CODE_END_PATTERN.sub('', line)

        tokens = line.split()
        name_tokens = []
        values = []
        found_first_value = False

        for token in tokens:
            if not found_first_value and not (self._is_valid_number(token) or token in self.MISSING_VALUES):
                name_tokens.append(token)
            else:
                found_first_value = True
                if self._is_valid_number(token):
                    values.append(token)
                elif token in self.MISSING_VALUES:
                    values.append('.')

        return ' '.join(name_tokens), values

    def _process_even_page(self, text: str) -> Dict[str, FoodRecord]:
        """Procesa una página par (código al inicio)."""
        records = {}

        for line in text.split('\n'):
            code = self._extract_code(line, 'start')
            if not code:
                continue

            # Remover código del inicio
            line_without_code = self.CODE_START_PATTERN.sub('', line.strip())
            name, values = self._process_line_tokens(line_without_code)

            records[code] = FoodRecord(codigo=code, nombre=name, valores=values)

        return records

    def _process_odd_page(self, text: str, existing_records: Dict[str, FoodRecord]) -> None:
        """Procesa una página impar (código al final) actualizando records existentes."""
        for line in text.split('\n'):
            code = self._extract_code(line, 'end')
            if not code or code not in existing_records:
                continue

            # Extraer valores excluyendo el código
            _, values = self._process_line_tokens(line, skip_code=True)
            existing_records[code].valores.extend(values)

    def extract_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Extrae datos del PDF y retorna DataFrame directamente."""
        combined_data: Dict[str, FoodRecord] = {}

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            # Procesar en lotes para mejor uso de memoria
            logger.info(f"Procesando {total_pages} páginas...")

            # Primero todas las páginas pares
            for page_num in range(2, total_pages + 1, 2):  # 2, 4, 6...
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                if text:
                    page_records = self._process_even_page(text)
                    combined_data.update(page_records)

            logger.info(f"Páginas pares procesadas: {len(combined_data)} registros encontrados")

            # Luego todas las páginas impares
            for page_num in range(1, total_pages + 1, 2):  # 1, 3, 5...
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                if text:
                    self._process_odd_page(text, combined_data)

            logger.info("Páginas impares procesadas")

        # Convertir a DataFrame directamente
        return self._build_dataframe(combined_data)

    def _build_dataframe(self, records: Dict[str, FoodRecord]) -> pd.DataFrame:
        """Construye DataFrame optimizado."""
        # Pre-allocar lista para mejor rendimiento
        data_list = []

        for record in records.values():
            row = ['NaN'] * len(self.columns)
            row[0] = record.codigo
            row[1] = record.nombre

            # Asignar valores con manejo de índices
            for i, val in enumerate(record.valores):
                if i < len(self.columns) - 2:
                    row[i + 2] = '.' if val == '.' else val.replace(',', '.')

            data_list.append(row)

        # Crear DataFrame de una vez
        df = pd.DataFrame(data_list, columns=self.columns)

        # Log de estadísticas
        self._log_statistics(df)

        return df

    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Registra estadísticas del DataFrame."""
        data_cols = self.columns[2:]
        total_cells = len(df) * len(data_cols)

        # Usar operaciones vectorizadas de pandas
        nan_mask = df[data_cols] == 'NaN'
        dot_mask = df[data_cols] == '.'

        nan_count = nan_mask.sum().sum()
        dot_count = dot_mask.sum().sum()
        numeric_count = total_cells - nan_count - dot_count

        logger.info(f"\nEstadísticas de extracción:")
        logger.info(f"├─ Total registros: {len(df)}")
        logger.info(f"├─ Valores numéricos: {numeric_count:,} ({numeric_count/total_cells*100:.1f}%)")
        logger.info(f"├─ Valores faltantes ('.'): {dot_count:,} ({dot_count/total_cells*100:.1f}%)")
        logger.info(f"└─ Valores no capturados (NaN): {nan_count:,} ({nan_count/total_cells*100:.1f}%)")

        # Columnas con más datos
        numeric_per_col = (~(nan_mask | dot_mask)).sum()
        top_cols = numeric_per_col.nlargest(5)
        logger.info(f"\nTop 5 columnas con más datos:")
        for col, count in top_cols.items():
            logger.info(f"  - {col}: {count} valores")

    def process_and_save(self, pdf_path: str, output_path: str) -> pd.DataFrame:
        """Proceso principal: extrae del PDF y guarda CSV."""
        logger.info(f"Iniciando extracción de: {pdf_path}")

        # Extraer datos
        df = self.extract_from_pdf(pdf_path)

        # Guardar CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✓ CSV guardado en: {output_path}")

        return df

# Función de conveniencia para uso directo
def extract_nutrition_data(pdf_path: str, output_path: str) -> pd.DataFrame:
    """Función principal para extraer datos nutricionales de PDF a CSV."""
    extractor = OptimizedNutritionExtractor()
    return extractor.process_and_save(pdf_path, output_path)

# Ejemplo de uso
if __name__ == "__main__":
    # Uso simple
    df = extract_nutrition_data(
        pdf_path="./data/tablas-peruanas-QR.pdf",
        output_path="./data/nutricion_completa.csv"
    )

    # Opcional: análisis adicional
    print(f"\nPrimeros 5 registros:")
    print(df.head())

    # Ver registros más completos
    mask = (df[df.columns[2:]] != 'NaN').sum(axis=1) > 20
    if mask.any():
        print(f"\nRegistros más completos:")
        print(df[mask].iloc[:3])
