import gzip
import matplotlib.pyplot as plt
import argparse

def parse_tsp(file_path):
    """Parsea un archivo TSP y devuelve metadatos y ciudades"""
    metadata = {}
    cities = []
    with gzip.open(file_path, 'rt') as f:
        capture = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                capture = True
                continue
            if line == "EOF":
                break
            if capture:
                parts = line.split()
                cities.append((float(parts[1]), float(parts[2])))
            else:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    return metadata, cities

def parse_tour(file_path):
    """Parsea un archivo de tour y devuelve metadatos y el tour"""
    metadata = {}
    tour = []
    header_lines = []
    with gzip.open(file_path, 'rt') as f:
        capture = False
        for line in f:
            clean_line = line.strip()
            if clean_line == "TOUR_SECTION":
                header_lines.append(line.strip())  # Conservar formato original
                capture = True
                continue
            if clean_line in ("-1", "EOF"):
                header_lines.append(clean_line)  # Añadir marcas finales
                break
            if capture:
                try:
                    tour.append(int(clean_line))  # Mantener números originales
                    header_lines.append(clean_line)  # Añadir línea al tour
                except ValueError:
                    pass
            else:
                if ':' in clean_line:
                    key, value = clean_line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                header_lines.append(clean_line)
    return metadata, tour, header_lines

def print_tour_info(header_lines, tour):
    """Imprime la información del tour en formato legible"""
    print("\nTOUR INFORMATION:")
    # Imprimir cabecera completa
    for line in header_lines:
        if line == "TOUR_SECTION":
            print(line)
            # Mostrar elementos
            for num in tour:
                print(num)
            print("-1")
            print("EOF")
            break
        print(line)

def visualize(tsp_file, tour_file=None):
    """Función principal de visualización"""
    # Procesar archivo TSP
    tsp_metadata, cities = parse_tsp(tsp_file)
    x = [c[0] for c in cities]
    y = [c[1] for c in cities]

    # Configurar gráfico
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c='red', s=50, label='Ciudades')

    # Procesar tour si existe
    if tour_file:
        tour_metadata, tour, header_lines = parse_tour(tour_file)
        print_tour_info(header_lines, tour)
        
        # Convertir a índices 0-based para el plotting
        tour_indices = [i-1 for i in tour]
        tour_path = [cities[i] for i in tour_indices]
        tour_path.append(tour_path[0])  # Cerrar el ciclo
        
        tx = [p[0] for p in tour_path]
        ty = [p[1] for p in tour_path]
        plt.plot(tx, ty, 'b-', linewidth=1, alpha=0.6, label='Ruta ACO')

    # Mostrar gráfico
    plt.title(f'TSP: {tsp_metadata.get("NAME", "Desconocido")}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizador de problemas TSP y soluciones ACO')
    parser.add_argument('tsp_file', help='Archivo .gz con el problema TSP')
    parser.add_argument('--tour_file', help='Archivo .gz con la solución de tour (opcional)')
    args = parser.parse_args()

    visualize(args.tsp_file, args.tour_file)