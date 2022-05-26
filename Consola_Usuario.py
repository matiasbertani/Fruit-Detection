from cmd import Cmd
import os
from timeit import default_timer


from Clasificacion_kmeans import  Reconocer_Fruta_kmneas, kmeans
import os

import cv2

import Fruta
import numpy as np
from base_datos import Base_Datos
from Clasificacion_knn import Knn
import imutils


class Interfaz_Usuario(Cmd):

    BASE_DATOS = None
    CENTROIDES = None

    def do_ENTRENAR(self, args):
        """Lee las imagenes de entremiento y las almacena en una base de datos, realizando el proceso de vision artificial """
        print('Leyendo imagenes de entrenamiento....\n\n')

        

        list_frutas = list()
        
        
        
        os.chdir('./img_entrenamiento/')
        actual = os.getcwd()
        imagenes = os.listdir()
        
        for img in imagenes:
            
            
            fruit = Fruta.Fruta()
            fruit.Leer_img(actual, img)
            fruit.Calcular_Caracteristicas()

            list_frutas.append(fruit)
    
        del(fruit)
        os.chdir('../')

        print('Lectura completa\n')    

        self.BASE_DATOS = Base_Datos(list_frutas)
        self.BASE_DATOS.Armar_Matriz_Caracterristicas()
        self.BASE_DATOS.Normalizar_Datos()
        
        print('\nIniciando CLASIFICACION KMEANs....\n\n')
        
        kmeans_aciertos =  int()
        #repara centroides y desplegarlos
        self.CENTROIDES = kmeans(self.BASE_DATOS)
        
        for fruta in list_frutas:
            print(f"La fruta es <-- {fruta.NOMBRE} -->, y kmeans dice : ---- {fruta.KMEANS_CLASIFICACION} ---- ")
            if fruta.KMEANS_CLASIFICACION == fruta.NOMBRE:
                kmeans_aciertos += 1
        del(fruta)
        kmeans_eficiencia = float(kmeans_aciertos / len(list_frutas))
        print(f'La eficiencia del K_MEANS es de: {kmeans_eficiencia}')

    def do_TESTING(self, args):
        """Comienza el proceso de testeo sobre imagenes de prueba, verificando la eficiencia de los metodos k_means y knn"""
        if self.BASE_DATOS is not None:
                print('\nIniciando CLASIFICACION KNN....\n\n')
                frutas_entrenemiento = list()
                knn_aciertos =  int()

                os.chdir('./testing_img/')
                actual = os.getcwd()
                
                imagenes = os.listdir()


            # --- PROCESO KNN sobre TEST_IMG
                inicio_knn = default_timer()
                for img in imagenes:
                        
                    fruit = Fruta.Fruta()
                    fruit.Leer_img(actual, img)
                    fruit.Calcular_Caracteristicas()

                    frutas_entrenemiento.append(fruit)
                    resultado = Knn( 5 , self.BASE_DATOS,  fruit)
                    fruit.KNN_CLASIFICACION = resultado
                    print(f"{fruit.Nombre_Archivo}: La fruta es <- {fruit.NOMBRE.upper()} ->, y knn dice : <- {fruit.KNN_CLASIFICACION.upper()} -> ")
                    if fruit.KNN_CLASIFICACION == fruit.NOMBRE:
                        knn_aciertos += 1
                fin_knn = default_timer()
                knn_eficiencia = round(float(knn_aciertos / len(frutas_entrenemiento)) *100,2)
                os.chdir('../')
                
                print(f'\n\n---La eficiencia del KNN es de: {knn_eficiencia}%---')
                print(f'El tiempo de clasificacion KNN tomo: {fin_knn- inicio_knn}\n')
            
            # --- PROCESO KMEANS sobre TEST_IMG  
                inicio_kmean = default_timer()
                kmeans_aciertos = 0
                for img in frutas_entrenemiento:
                    #Generar distancias a los distintos centrodes
                
                    Reconocer_Fruta_kmneas(img, self.CENTROIDES)
                    print(f"{img.Nombre_Archivo}: La fruta es <- {img.NOMBRE.upper()} ->, y KMEANS dice : <- {img.KMEANS_CLASIFICACION.upper()} -> ")                    
                    if img.KMEANS_CLASIFICACION == img.NOMBRE:
                        kmeans_aciertos += 1

                del(img)
                kmeans_eficiencia =round( float(kmeans_aciertos / len(frutas_entrenemiento))*100,2)
                fin_kmean = default_timer()

                print(f'\n\n ---La eficiencia del K_MEANS es de: {kmeans_eficiencia}% ---')
                print(f'El tiempo de clasificacion KMEANS tomo: {round(fin_kmean- inicio_kmean,2)} segundos\n')

        else:
            print('NO puede realizarse ningun pruaba de clasificacion ya que no se ha leido la Base de Entrenamiento')

    def do_STREAMING(self, args):
        """Inicia una ventana de video para realizar una clasificacion en streaming"""
        MODO = args
        if self.BASE_DATOS is not None:

            kernel_close = np.ones((9,9), np.uint8) 
            cap = cv2.VideoCapture(2)
            hay_contorno = bool()
            fruit = Fruta.Fruta()        
                
            i = 0

            while True:
                ret, frame = cap.read()

                if ret == False: break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if i == 20:
                    bgGray = gray
                
                if i > 20:
                    hay_contorno = False
                    dif = cv2.absdiff(gray, bgGray)
                    # cv2.imshow('dif',dif)
                    _, th = cv2.threshold(dif, 20, 255, cv2.THRESH_BINARY)
                    
                    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,kernel_close)
                    cv2.imshow('th',th)
                    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        
                        cnt = max(cnts, key=cv2.contourArea)
                                                    
                        area = cv2.contourArea(cnt)
                        area_frame = gray.shape[1] * gray.shape[0]

                        if  area_frame *0.01 < area and area < area_frame *0.8:
                            hay_contorno = True
                            img_aux = frame.copy()
                            x,y,w,h = cv2.boundingRect(cnt)
                            cv2.drawContours(frame, [cnt], -1, (0,0,255),2)
                            cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10),(0,255,0),2)                        

                                
                i = i+1
                if hay_contorno :
                    """
                    1 - recortar la imagen que me sirva
                    2 - imagen guardarla o leer en una variable fruta
                    3 - extraer sus caracteristicas
                    4 - aplicar K_means y K_nn y ver resultados
                    5 escribir en imagen frame el resutado
                    """
                    
                    # img_fruta = cv2.resize(frame[y:y + h, x:x + w], (100, 50))
                    try:
                        img_fruta = imutils.resize(img_aux[y-10:y+ 8 + h , x-10 :x + 8 + w ],width=300)
                        nombre = f'Imagen-Streamging_1.jpg'
                                    
                        fruit.GUardar_Img_Procesada(img_fruta, nombre)
                        try:
                            fruit.Calcular_Caracteristicas()
                            fruit.Mostrar_Img_Enmascarada()
                            fruit.KNN_CLASIFICACION  = Knn( 5 , self.BASE_DATOS,  fruit)
                            Reconocer_Fruta_kmneas(fruit, self.CENTROIDES)
                            cv2.putText(frame,f'KMEANS: {fruit.KMEANS_CLASIFICACION.upper()}',(x,y-40),2,0.7,(0,0,255),2,cv2.LINE_AA)

                            cv2.putText(frame,f'KNN: {fruit.KNN_CLASIFICACION.upper()}',(x,y-20),2,0.7,(0,0,255),2,cv2.LINE_AA)
                            # print(f"{fruit.Nombre_Archivo}: La fruta es <- {fruit.NOMBRE.upper()} ->, y knn dice : <- {fruit.KNN_CLASIFICACION.upper()} -> ")
                        except:
                            cv2.putText(frame,'NO SE RECONOCE LA FRUTA',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
                            
                            pass
                    except:
                        pass
                cv2.imshow('Frame',frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            

        else:
            print('NO PUEDE realizarse VIDEO STREAMING de clasificacion ya que no se ha Entrenado el Sistema')

        print('straeming')

    def default(self,args):
        print("Error. El comando \'" + args + "\' no existe")
               
    def precmd(self,args):
        
        if args.lower() == 'help':
            return args.lower() 
        else:
            return args.upper()
        
    def do_EXIT(self,args):
        """quit sale del interprete"""
        print("Sistema Cerrado")
        
        raise SystemExit

if __name__ == '__main__':
    
    interfaz = Interfaz_Usuario()
    interfaz.prompt = '>> '
        
    interfaz.cmdloop("Iniciando ventana de comandos...")
