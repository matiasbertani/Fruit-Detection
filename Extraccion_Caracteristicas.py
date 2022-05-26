from math import pi
import cv2
import imutils
import numpy as np
from numpy.lib.type_check import imag
import matplotlib.pyplot as plt

from collections import Counter

import Proseacimiento


def Get_Vector_Caracteristicas(imagen, mascara, contorno):

    CARACTERISTICAS = list()

    # cv2.imshow('Imagen Original', imagen)
    prop_cnt = Calc_Propiedades_Contorno(contorno,mascara)    

    prop_hist = Propiedades_Histograma(imagen, mascara)
    
 # !!!!!!!!!!!!!!!1  Usar concatencion de dos lista o empaquetamiento para unirlas y crear las caracteristicas
    for cnt in prop_cnt:
        CARACTERISTICAS.append(cnt)
    for hist in prop_hist:
        CARACTERISTICAS.append(hist)

    
    return CARACTERISTICAS

def Calc_Propiedades_Contorno( cnt , masc):
    """ 
    En principio recibe una Fruta, puede recibir el contorno de la misma directamente.
    calcula las propiedades mas relevante
    Las guarda en un LISTA  o en un DICCIONARIO (decidirlo)

    """
    propiedades = list()
    

 #MOMENTOS HU 
    x, y, width, height = cv2.boundingRect(cnt)

    roi = cv2.resize(masc[y:y + height, x:x + width], (50, 50))
    momentos_hu = cv2.HuMoments(cv2.moments(roi)).flatten()
    propiedades.extend([momentos_hu[0], momentos_hu[2], momentos_hu[2]]) #Usa los 3 primeros momentos
    # propiedades.extend(momentos_hu[0:3]) #Usa los 3 primeros momentos

    # MOMENTOS deL controno
    M = cv2.moments(cnt)
    
    
    #centroide    
    #Coordenadas del centoride
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
  

  #El AREA DEL CONTORNOP viene dada por la función cv2.contourArea() o por el momento M [‘m00’].
    area = cv2.contourArea(cnt) 

  # PERIMETRO
    perimetro = cv2.arcLength(cnt,True)
   

  # RELACION DE ASPECTO
    #razón entre el ancho y la altura del contorno del objeto
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h

  #LA EXTEBSION : es la razón entre el área del contorno y el área del rectángulo delimitador    
    rect_area = w*h   
    extension = float(area)/rect_area

  # SOLIDEZ
    #  es la razón entre el área del contorno y el área de su envoltura convexa.
    envoltura = cv2.convexHull(cnt)
    area_envoltura = cv2.contourArea(envoltura)
    solidez = float(area)/area_envoltura
  #   
    

    #Diámetro equivalente diámetro del círculo cuya área es igual que el área del contorno.
    equi_diametro = np.sqrt(4*area/np.pi)
    

    # print(f'El DIAMETRO EQUIVALENTE es : {equi_diametro}')




    # ORIENTACION
    # La orientación es el ángulo que forma el eje mayor de la elipse circunscrita  
    # al objeto, con la dirección horizontal. 
    # El siguiente método también da las longitudes del Eje Mayor y del Eje Menor de dicha elipse.
    # (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)   
    # print(f'La ORIENTACION  es : {angle}')
    
    #    Valores mínimo y máximo y sus respectivas coordenadas
    #   Estos valores pueden encontrarse utilizando una máscara de la imagen:
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
   
    #!!! propiedades.append(cx)
    #!!! propiedades.append(cy)
   
    #!!! propiedades.append(area)
    # propiedades.append(aspect_ratio)
    #!!! propiedades.append(perimetro)
    #!!!--< propiedades.append(equi_diametro) ------ lo use y bajo considerablemente la eficiencia
    #!!!--< propiedades.append(rect_area)
    # propiedades.append(extension)
   
    # propiedades.append(solidez)
   

    compacidad = (perimetro**2) / area
    redondez = (4 * pi * area) / (perimetro**2)
    propiedades.append(redondez)
    propiedades.append(compacidad)
    

    return propiedades




def Propiedades_Histograma(img, masc):
    """
    toma la imagen y calcula el valor maximo de  histograma
    """

    propiedades = list()

    #Histograma en BGR
    color = ('b','g','r')
    max_bgr = list()
    for i, col in enumerate(color):
        
        histr = cv2.calcHist( [img], [i], masc, [255], [0,255])
        max_bgr.append( np.argmax( histr ) )
        
    #     plt.plot( histr, color = col )
    # plt.show()



    #Histograma en HSV
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist =  cv2.calcHist([im_hsv], [0], masc, [180], [ 0, 180])  #[180, 256], [0, 180, 0, 256])
    # plt.plot( hist, color = 'g' )
     

    max_h = np.argmax(hist)
    
    #se podria agragar varios maximos 4 o mas comunes con counter. pero tendriamos que usar histogrmaa hecho con numpy no con opencv
    
    # print(f"El valor MAXIMO de H es : {max_h}")
        
    # plt.show()
    # # contornos.Aplicar_Mascara(img,masc)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

#  #AGREGA BGR
#     for max in max_bgr:
#         propiedades.append(max)

 # AGREGAR H
    propiedades.append(max_h)
    return propiedades




if __name__ == "__main__":

    CARACTERISTICAS = list()

    imagen = cv2.imread("img/limon_6.jpg")
    imagen = cv2.resize(imagen, (230,230))
    imagen = imutils.resize(imagen, width= 230)

    # cv2.imshow('Imagen Original', imagen)
    mascara , contorno = Proseacimiento.Filtrados(imagen)
    prop_cnt = Calc_Propiedades_Contorno(contorno, mascara)
    print('Cantidad de Proiedades de CONTORNO: ', len(prop_cnt))

    prop_hist = Propiedades_Histograma(imagen, mascara)
    print('Cantidad de Proiedades de HISTOGRAMA: ', len(prop_hist))
    
 # !!!!!!!!!!!!!!!1  Usar concatencion de dos lista o empaquetamiento para unirlas y crear las caracteristicas

    for cnt in prop_cnt:
        CARACTERISTICAS.append(cnt)
    for hist in prop_hist:
        CARACTERISTICAS.append(hist)

    print(CARACTERISTICAS)