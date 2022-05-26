import random
from statistics import mode
from base_datos import Base_Datos

from typing import List
from Fruta import Fruta

"""
ALGORITMO K_MEANS
-----------------


Luego del preprocesamientos, filtrados, y segmentacion y extraccion de caracteristicas.

BASE DE DATOS: Lista de imagenes de frutas

    COMIENZA LA CLASIFICACION

        1- Toma la base de datos.
            (Clase base de datos)
            posee lista con referencia a cada img de fruta
            posee dos vectores de caracteriasticas uno con masximo y otro con minimos para la generacions 
        
        2- Genera "Frutas" Aleatorias --- centroides, serian 4 
            - Copn los min y max, genera puntos al azar en ese rango
        
        3- 

El Vector de caracteristicas es basicamente son las coordenadas n-dimensional.
"""
def Reconocer_Fruta_kmneas(img,centroides):
    """recibe Una imagen y centroides y clasifica la imagne"""

    lista_deltas = list()
                
    for centro in centroides:
        delta_temp = img.Cacular_Distancia_Vector(centro)
        lista_deltas.append(delta_temp)
        
        
    #Calcular el minimo del centroide, obtener su indce o posiscion en la lista 
    
    ind = lista_deltas.index(min(lista_deltas))
    img.KMEANS_CLASIFICACION = centroides[ind].KMEANS_CLASIFICACION

    


def Generar_Centroide(cluster) -> Fruta():
    """Recibe una lista de clasificacion y entrega su centroide"""
    
    centro = Fruta()

    # Crea el vector centroide vacio
    coordenadas_centro = list()
    for i in range(len(cluster[0].VECTOR_CARACTERISTICAS)):
        coordenadas_centro.append(0)

    it = 0
    
    # para cada fruta, lo adiciona
    for fruta in cluster:
        

        it += 1
        vec_prop = fruta.VECTOR_CARACTERISTICAS
        
        for i , prop in enumerate(vec_prop):
            coordenadas_centro[i] += prop

    for i, corde in enumerate(coordenadas_centro):
        coordenadas_centro[i] = corde/len(cluster)

    centro.Set_Vector_Caracteristica(coordenadas_centro)

    return centro


def Generar_Lista_Centroides(base)-> list():
    """Recibe un nuemro de centroides y genera los mismos en una lista"""
    nombres = ['banana_3.jpg','limon_1.jpeg','naranja_7.jpg','tomate_1.jpg']
    centroides = list()
    
    for i in nombres:
        centro = Fruta()
        for fruti in base.FRUTAS:
            
            if fruti.Nombre_Archivo == i:
                centro = fruti
                centroides.append(centro)
        
  
    return centroides

def kmeans(base_datos, grupos = 4):
    
    LISTA_FRUTAS = base_datos.FRUTAS 
   
    CENTROIDES = Generar_Lista_Centroides( base_datos)
    print('\nLos centroides Inicialies son :')
    for i in CENTROIDES:
        print(i.Nombre_Archivo)
        print(i.VECTOR_CARACTERISTICAS)
    CLASIFICACION = list() #lista de los grupos o clusters 
    for i in range(grupos):
        CLASIFICACION.append(list())


    iterador = 0    
    itmax = 50000

    convergencia = False
    it_converg = 0
    it_converg_MAX = 1000
    
    while not convergencia and iterador < itmax: # mientras no converja o pase un numero exesivo de iteraciones
        
        
        clasificacion_temporal = list() # lista de los cluster de clasificacion temproal (lista de listas)
        
        #genera la lista de acuerdo a cuantos centroides tengamos 4 centroides 4 equipos
        for i in range(len(CENTROIDES)): 
            
            clasificacion_temporal.append(list())
        
        #Comienza 
        itito =0
        for una_fruta in LISTA_FRUTAS:
            
            lista_deltas = list()
           
            #Generar distancias a los distintos centrodes
            for centroide in CENTROIDES:
                delta_temp = una_fruta.Cacular_Distancia_Vector(centroide)
                lista_deltas.append(delta_temp)
            
            
            #Calcular el minimo del centroide, obtener su indce o posiscion en la lista 
            min_delta = min(lista_deltas)
            ind = lista_deltas.index(min_delta)

            equipo = f'GRUPO {ind}' # nos e si suar esto, o usarlo para modificiar el tipo en la fruta misma

            clasificacion_temporal[ind].append(una_fruta)
            itito += 1
            #asignar a fruta a la clasificacion obtenida
            
            

        #Creacion de nuevos cenrtoides a los nuevos grupos                
        nuevos_centroides = list()

        for i, centroide in enumerate(CENTROIDES):
            
            new_centroide = Fruta()  #tiene que tener ciertos parametros para iniciar vacio

            if clasificacion_temporal[i]:
                new_centroide = Generar_Centroide(clasificacion_temporal[i])
            else:
                print('El programa mismo centroide')
                new_centroide = centroide

            nuevos_centroides.append(new_centroide)
            del(new_centroide)
        
        flag_converg = 0
        for i,new_cent in enumerate(nuevos_centroides):
            
            if new_cent.VECTOR_CARACTERISTICAS == CENTROIDES[i].VECTOR_CARACTERISTICAS:
                flag_converg += 1
            

        # Control de convergencia
        if flag_converg == 4:

            it_converg += 1

            if it_converg > it_converg_MAX:
                convergencia = True
                print('El Algoritmo convergio')
        # for i,centr in enumerate(CENTROIDES):
        #     print(f'Centroide de Grupo: {i+1}')
        #     print(centr.VECTOR_CARACTERISTICAS)

        #Actualizacion de variables
        for  i, new_centr  in  enumerate(nuevos_centroides):
            CENTROIDES[i] = new_centr

        for i, cluster in enumerate(clasificacion_temporal):
            CLASIFICACION[i] = cluster
                
        del(clasificacion_temporal)
        
        #Control de iteracion
        iterador += 1


    print('\nAlgoritmo de clasificacion terminado')
    print(f'Cantidad de Iteraciones: {iterador}')
    print(f'La maxima cantidad de converghencia fue de {it_converg}')
    print('\nLos Centroides resultado son:\n')
  
    for i,centr in enumerate(CENTROIDES):
        print(f'Centroide de Grupo: {i+1}')
        print(centr.VECTOR_CARACTERISTICAS)
    for i , cluster in enumerate(CLASIFICACION):
        
        nombres = list()
        for fruta in cluster:
            nombres.append(fruta.NOMBRE)
        resultado = mode(nombres)
        CENTROIDES[i].KMEANS_CLASIFICACION = resultado 
        
        del(fruta)
        print(f'\n\nGRUPO {i+1} : {resultado}')
        for fruta in cluster:
            fruta.KMEANS_CLASIFICACION = resultado
            print(fruta.Nombre_Archivo)

    return CENTROIDES
            
    


if __name__ == "__main__":
    pass

