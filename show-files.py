import os
import sys

def mostrar_arbol(ruta, prefijo="", es_ultimo=True, es_raiz=True):
    """
    Muestra la estructura de carpetas en formato árbol
    
    Args:
        ruta: Ruta del directorio a mostrar
        prefijo: Prefijo para la indentación
        es_ultimo: Si es el último elemento del directorio actual
        es_raiz: Si es el directorio raíz
    """
    try:
        nombre = os.path.basename(ruta)
        
        if es_raiz:
            print(f"{nombre}/")
        else:
            conector = "└── " if es_ultimo else "├── "
            sufijo = "/" if os.path.isdir(ruta) else ""
            print(f"{prefijo}{conector}{nombre}{sufijo}")
        
        if os.path.isdir(ruta):
            try:
                elementos = sorted(os.listdir(ruta))
                # Filtrar elementos ocultos si lo deseas (opcional)
                # elementos = [e for e in elementos if not e.startswith('.')]
                
                for i, elemento in enumerate(elementos):
                    ruta_completa = os.path.join(ruta, elemento)
                    es_ultimo_elemento = (i == len(elementos) - 1)
                    
                    if es_raiz:
                        nuevo_prefijo = ""
                    else:
                        extension = "    " if es_ultimo else "│   "
                        nuevo_prefijo = prefijo + extension
                    
                    mostrar_arbol(ruta_completa, nuevo_prefijo, es_ultimo_elemento, False)
                    
            except PermissionError:
                print(f"{prefijo}    [Acceso denegado]")
                
    except Exception as e:
        print(f"Error al procesar {ruta}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python tree_folder.py <nombre_carpeta>")
        print("Ejemplo: python tree_folder.py backend")
        sys.exit(1)
    
    carpeta = sys.argv[1]
    
    # Verificar si la carpeta existe
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe")
        sys.exit(1)
    
    if not os.path.isdir(carpeta):
        print(f"Error: '{carpeta}' no es una carpeta")
        sys.exit(1)
    
    # Mostrar el árbol
    mostrar_arbol(carpeta)


if __name__ == "__main__":
    main()