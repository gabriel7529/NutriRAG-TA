import pdfplumber
import re
import csv
import os
from typing import List, Dict, Optional, Tuple

class ExtractorCENANFixed:
    def __init__(self):
        self.debug = True

    def clean_duplicate_chars(self, text: str) -> str:
        """Limpia caracteres duplicados del PDF"""
        if not text:
            return ""
        cleaned = re.sub(r'([A-ZÁÉÍÓÚÑ])\1+', r'\1', text)
        return cleaned.strip()

    def extract_nutrition_from_pair_page(self, text: str) -> Dict[str, float]:
        """Extrae valores nutricionales de páginas pares"""
        nutrition = {'energia': 0.0, 'proteinas': 0.0, 'hierro': 0.0}

        energia_match = re.search(r'(\d+)\s*kcal', text)
        if energia_match:
            nutrition['energia'] = float(energia_match.group(1))

        proteinas_match = re.search(r'([\d,\.]+)\s*g', text)
        if proteinas_match:
            value = proteinas_match.group(1).replace(',', '.')
            nutrition['proteinas'] = float(value)

        hierro_match = re.search(r'([\d,\.]+)\s*mg', text)
        if hierro_match:
            value = hierro_match.group(1).replace(',', '.')
            nutrition['hierro'] = float(value)

        return nutrition

    def extract_title_from_odd_page(self, text: str) -> str:
        """Extrae título de páginas impares (más limpio)"""
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Buscar líneas que sean títulos: mayúsculas, longitud razonable
            if (len(line) > 5 and len(line) < 50 and
                line.isupper() and
                not any(word in line for word in
                       ['REPÚBLICA', 'MINISTERIO', 'PERÚ', 'INSTITUTO',
                        'INGREDIENTES', 'PREPARACIÓN', 'RACIONES', 'SALUD'])):
                return line.title()
        return ""

    def clean_ingredient(self, ingredient: str) -> str:
        """Limpia un ingrediente individual eliminando texto de preparación"""
        # Eliminar palabras que claramente pertenecen a preparación
        prep_words = ['luego', 'freírlas', 'pimienta,', 'sazonados', 'cortado', 'cortada',
                     'pequeños', 'cuadritos', 'desmenuzarlo', 'caliente', 'anterior',
                     'condimentar', 'agregar', 'retirarlo', 'aceite', 'vegetal']

        words = ingredient.split()
        clean_words = []

        for word in words:
            # Parar si encontramos una palabra de preparación
            if any(prep_word in word.lower() for prep_word in prep_words):
                break
            clean_words.append(word)

        result = ' '.join(clean_words).strip()

        # Eliminar caracteres finales problemáticos
        result = re.sub(r'[,\.]$', '', result)

        return result

    def extract_ingredients_smart(self, text: str) -> str:
        """
        Extracción inteligente de ingredientes
        Busca líneas que empiezan con '-' y extrae solo la parte del ingrediente
        """
        ingredients = []
        lines = text.split('\n')

        print(f"DEBUG - Encontradas {len([l for l in lines if l.strip().startswith('-')])} líneas con ingredientes")

        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()

            # Solo procesar líneas que empiezan con -
            if line.startswith('-'):
                # Remover el guión inicial
                ingredient_text = line[1:].strip()

                # Método 1: Buscar hasta el primer número seguido de punto
                match = re.search(r'^(.*?)(?=\s*\d+\.)', ingredient_text)
                if match:
                    ingredient = match.group(1).strip()
                else:
                    # Método 2: Tomar solo las primeras 6-8 palabras como ingrediente
                    words = ingredient_text.split()
                    if len(words) > 6:
                        ingredient = ' '.join(words[:6])
                    else:
                        ingredient = ingredient_text

                # Validar que el ingrediente sea razonable
                if ingredient and len(ingredient) > 3 and len(ingredient) < 100:
                    # Limpiar el ingrediente
                    clean_ingredient = self.clean_ingredient(ingredient)
                    if clean_ingredient and len(clean_ingredient) > 3:
                        ingredients.append(clean_ingredient)

        result = ', '.join(ingredients)
        print(f"DEBUG - {len(ingredients)} ingredientes extraídos")
        return result

    def extract_preparation_smart(self, text: str) -> str:
        """
        Extracción inteligente de preparación
        Busca texto después de números seguidos de punto
        """
        preparation_parts = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # Buscar líneas que empiezan con número seguido de punto
            if re.match(r'\d+\.', line):
                preparation_parts.append(line)

            # También buscar en líneas que tienen "- ingrediente 1. preparacion"
            elif line.startswith('-') and re.search(r'\d+\.', line):
                # Extraer solo la parte después del número
                match = re.search(r'\d+\.\s*(.*)', line)
                if match:
                    prep_text = match.group(1).strip()
                    if prep_text:
                        preparation_parts.append(match.group(0))  # Incluir el número

        return ' '.join(preparation_parts)

    def extract_recipes_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extrae recetas del PDF con debugging mejorado"""
        recipes = []

        if not os.path.exists(pdf_path):
            print(f"❌ Archivo no encontrado: {pdf_path}")
            return recipes

        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"📖 Procesando PDF con {len(pdf.pages)} páginas...")

                # Procesar todas las recetas del PDF
                for i in range(11, len(pdf.pages) - 1, 2):  # Procesar todas las recetas
                    try:
                        pair_page = pdf.pages[i]
                        pair_text = pair_page.extract_text()

                        odd_page = pdf.pages[i + 1]
                        odd_text = odd_page.extract_text()

                        if not pair_text or not odd_text:
                            continue

                        print(f"\n--- PROCESANDO PÁGINAS {i+1}-{i+2} ---")

                        # Extraer información
                        title = self.extract_title_from_odd_page(odd_text)  # Cambió a página impar
                        nutrition = self.extract_nutrition_from_pair_page(pair_text)

                        print(f"Título: '{title}'")
                        print(f"Nutrición: E={nutrition['energia']} P={nutrition['proteinas']} H={nutrition['hierro']}")

                        # Solo hacer debugging detallado si hay problemas
                        if not title:
                            print(f"DEBUG - Líneas de página impar:")
                            for j, line in enumerate(odd_text.split('\n')[:10]):
                                print(f"  {j+1:2d}: {repr(line)}")

                        ingredients = self.extract_ingredients_smart(odd_text)
                        preparation = self.extract_preparation_smart(odd_text)

                        print(f"Ingredientes: {len(ingredients)} chars")
                        print(f"Preparación: {len(preparation)} chars")

                        # Solo guardar si tenemos datos completos
                        if title and ingredients and preparation and nutrition['energia'] > 0:
                            recipe = {
                                'nombre': title,
                                'ingredientes': ingredients,
                                'preparacion': preparation,
                                'energia_kcal': nutrition['energia'],
                                'proteinas_g': nutrition['proteinas'],
                                'hierro_mg': nutrition['hierro'],
                                'raciones': 4,
                                'descripcion': f'Receta nutritiva: {nutrition["energia"]} kcal'
                            }

                            recipes.append(recipe)
                            print(f"✅ GUARDADA: {title}")
                        else:
                            print(f"❌ INCOMPLETA: T={bool(title)} I={bool(ingredients)} P={bool(preparation)} E={nutrition['energia']>0}")

                    except Exception as e:
                        print(f"❌ Error en páginas {i+1}-{i+2}: {e}")
                        continue

        except Exception as e:
            print(f"❌ Error abriendo PDF: {e}")

        return recipes

    def save_to_csv(self, recipes: List[Dict], output_path: str = 'recetas_cenan_fixed.csv'):
        """Guarda las recetas en CSV"""
        if not recipes:
            print('⚠️ No hay recetas para guardar.')
            return

        fieldnames = ['nombre', 'ingredientes', 'preparacion', 'energia_kcal',
                     'proteinas_g', 'hierro_mg', 'raciones', 'descripcion']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recipes)

        print(f'✅ {len(recipes)} recetas guardadas en: {output_path}')

def main():
    """Función principal con debugging detallado"""
    pdf_path = './data/CENAN-0013.pdf'

    extractor = ExtractorCENANFixed()

    print('🚀 EXTRACTOR CENAN FIXED - DEBUGGING DETALLADO')
    print('=' * 70)

    recipes = extractor.extract_recipes_from_pdf(pdf_path)

    if recipes:
        print(f'\n🎉 ÉXITO! {len(recipes)} recetas extraídas')
        extractor.save_to_csv(recipes)

        print('\n📋 RECETAS EXTRAÍDAS:')
        for i, recipe in enumerate(recipes, 1):
            print(f"\n{i}. {recipe['nombre']}")
            print(f"   📊 {recipe['energia_kcal']} kcal")
            print(f"   🥘 Ingredientes: {recipe['ingredientes'][:80]}...")
    else:
        print('\n❌ No se extrajeron recetas')

if __name__ == '__main__':
    main()
