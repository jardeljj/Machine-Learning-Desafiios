import cv2
import numpy as np

def processar_imagem(caminho_imagem, limiar=127):
    # Carregar a imagem em modo colorido
    imagem_colorida = cv2.imread(caminho_imagem)

    if imagem_colorida is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return

    # Converter para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

    # Aplicar binarização (thresholding)
    _, imagem_binarizada = cv2.threshold(imagem_cinza, limiar, 255, cv2.THRESH_BINARY)

    # Mostrar imagens
    cv2.imshow("Imagem Cinza", imagem_cinza)
    cv2.imshow("Imagem Binarizada", imagem_binarizada)

    # Salvar as imagens processadas
    cv2.imwrite("imagem_cinza.png", imagem_cinza)
    cv2.imwrite("imagem_binarizada.png", imagem_binarizada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemplo de uso:
caminho = "(coloque o caminho da sua imagem)"  # Substitua pelo caminho correto da sua imagem
processar_imagem(caminho)

# As imagens na pasta foram criadas com o algoritmo acima.
