import cv2 as cv
import streamlit as st
from PIL import Image
import numpy as np
from skimage import morphology, filters


def brilho_imagem(imagem, resultado):
    return cv.convertScaleAbs(imagem, beta=resultado)

def borra_imagem(imagem, resultado):
    return cv.GaussianBlur(imagem, (1,1), resultado)

def melhora_detalhe(imagem): 
    return cv.detailEnhance(imagem, sigma_s=34, sigma_r=0.5)

def escala_cinza(imagem):
	return cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

def principal():

	st.title("OpenCV Data App")
	st.subheader("Esse aplicativo web permite integrar processamento de imagens com OpenCV")
	st.text("Streamlit com OpenCV")

	arquivo_imagem = st.file_uploader("Envie sua imagem", type=["jpg", "png", "jpeg", "jfif"])

	with st.sidebar:
		taxa_borrao = st.slider("Borrão", min_value=0.0, max_value=3.5)
		qtd_brilho = st.slider("Brilho", min_value=-50, max_value=50, value=0)
		filtro_aprimoramento = st.checkbox("Melhorar detalhes da imagem")
		img_cinza = st.checkbox("Converter para escala de cinza")
		img_erosao = st.checkbox("Filtro Erosão")
		img_dilatacao = st.checkbox("Filtro Dilatação")
		img_edge = st.checkbox("Filtro Edge")

	if not arquivo_imagem:
		return None

	imagem_original = Image.open(arquivo_imagem)
	imagem_original = np.array(imagem_original)

	imagem_processada = borra_imagem(imagem_original, taxa_borrao)
	imagem_processada = brilho_imagem(imagem_processada, qtd_brilho)

	if filtro_aprimoramento:
		imagem_processada = melhora_detalhe(imagem_processada)

	if img_cinza:
		imagem_processada = escala_cinza(imagem_processada)

	if img_erosao:
		imagem_processada = morphology.erosion(imagem_processada)

	if img_dilatacao:
		imagem_processada = morphology.dilation(imagem_processada)

	if img_edge:
		imagem_processada = filters.sobel(imagem_processada)

	st.text("Imagem Original vs Imagem Processada")
	st.image([imagem_original, imagem_processada])


if __name__ == '__main__':
	principal()