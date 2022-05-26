import math
from typing import overload
import cv2
import imutils

import Proseacimiento
import Extraccion_Caracteristicas

class Fruta(object):

    ANCHO_STANDAR = 300
   
    
    def __init__(self) -> None:
        
        self.NOMBRE = str() #dEDUCIDO DE NOMBRE DE ARCHIVO
        self.KMEANS_CLASIFICACION = str()
        self.KNN_CLASIFICACION = str()   


        self.ANCHO = int()
        self.ALTO =  int()
        
        self.TIPO = str()


        self.IMG_ORIGINAL = None
        self.IMG_GRIS = None

        self.CONTORNO = None                
        self.MASCARA = None
        
        self.historgrama_color = None

        self.VECTOR_CARACTERISTICAS = tuple() # o una lista
        self.NORM_VECTOR_CARACTERISTICAS = tuple()
        
    
    def Leer_img(self ,path, name):
        self.Nombre_Archivo = name
        
        ruta = path + '/' + name

        
        img_original = cv2.imread(ruta)
        #veamos si funcionea sin resize el codigo
        # self.IMG_ORIGINAL = cv2.resize(img_original, (self.ANCHO,self.ALTO),interpolation= cv2.INTER_AREA)
        self.IMG_ORIGINAL = imutils.resize(img_original, width= self.ANCHO_STANDAR)
        self.Calcular_Nombre_fruta()
        self.ANCHO = self.IMG_ORIGINAL.shape[1]
        self.ALTO = self.IMG_ORIGINAL.shape[0]

    def GUardar_Img_Procesada(self,img,nombre):
        
        self.Nombre_Archivo = nombre
        self.IMG_ORIGINAL = imutils.resize(img, width= self.ANCHO_STANDAR)
        self.Calcular_Nombre_fruta()
        self.ANCHO = self.IMG_ORIGINAL.shape[1]
        self.ALTO = self.IMG_ORIGINAL.shape[0]


    def Cacular_Distancia_Vector(self, otro):
        
        distancia = float()
        sum = float()
        for i in range(len(self.VECTOR_CARACTERISTICAS)):
            
            delta_carac = self.VECTOR_CARACTERISTICAS[i] - otro.VECTOR_CARACTERISTICAS[i]
            sum += delta_carac**2

        distancia = math.sqrt(sum)
        return distancia

    def Set_Tipo(self,tipo):
        self.TIPO = tipo

    def Set_Vector_Caracteristica(self, carac):
        self.VECTOR_CARACTERISTICAS = carac

    def Set_Imagenes_Original( self , path,name ):
        
        self.Nombre_Archivo = name
        
    def Mostrar_img(self):
        cv2.imshow(self.Nombre_Archivo , self.IMG_ORIGINAL)

    def Mostrar_Img_Enmascarada(self):
        
        res = cv2.bitwise_and(self.IMG_ORIGINAL,self.IMG_ORIGINAL,mask = self.MASCARA)
        cv2.imshow('Enmascaramiento',res)

    def Calcular_Nombre_fruta(self):
        name = self.Nombre_Archivo
        nombre_real = name[:name.find('_')]
        self.NOMBRE = nombre_real

    def Calcular_Caracteristicas(self, MODO = 1):
        img = self.IMG_ORIGINAL.copy()
        
        if MODO == 1:
            self.MASCARA, self.CONTORNO = Proseacimiento.Filtrados(img) 
        elif MODO == 2: 
            self.MASCARA, self.CONTORNO = Proseacimiento.PreProcesamiento(img) 
        cnt = self.CONTORNO.copy()
        masc = self.MASCARA.copy()
        
        # contornos.Aplicar_Mascara(img,masc)
        
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        caracteristicas = Extraccion_Caracteristicas.Get_Vector_Caracteristicas( img, masc, cnt )    
       
        
        self.Set_Vector_Caracteristica(caracteristicas)

    def __str__(self) -> str:
        
        impresion = '['
        for carac in self.VECTOR_CARACTERISTICAS:
            impresion += f' {carac},'
        
        impresion += ']'
        return impresion


if __name__ == "__main__":
    print('hola')
    # fruta = Fruta('parche', 'pachi')
    # print(fruta)

    centro = Fruta.Centroide()
    print(type(centro))
    print(centro.TIPO)


