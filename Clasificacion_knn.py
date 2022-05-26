import pandas as pd
from statistics import mode

from Fruta import Fruta
from base_datos import Base_Datos

def Knn( ka , base  , fruta = Fruta()):
    """
    Recibe: 
        - un K
        - una lista que posee los clusters o grupos de clasificacion
        - el punto que se queire clasificar
    
    Entrega:
        - El resultado del equipo de eprtenencia del punto
    """

    """ 
    Armar un DataFrame incluyendo la distancia a cada Fruta
    -Ordenar el Data_Frame de menos distancia a mas distnacia
    
    """
    

    distancia = float()        
    q_azul = int()
    q_rojo = int()
    
    datos_puntos = list()
    df_knn = base.Get_df_knn(fruta)

  
    df_knn.sort_values( by= 'DISTANCIA', inplace = True, ascending = True )
    df_knn.reset_index( inplace= True )
    # print(df_puntos)

    frutas_cercanas = list()

    for i in range( 0, ka ):
        
        tipo = df_knn.at[ i , 'NOMBRE' ]   
        frutas_cercanas.append(tipo)

        if tipo == 'ROJO':
            q_rojo += 1
        
        if tipo == 'AZUL':
            q_azul += 1
    # print(frutas_cercanas)
    # print('El punto mas frecuente es: ',Mas_Frecuente(frutas_cercanas))
    # print('Azul: ',q_azul)
    # print('Rojo: ',q_rojo)

    Mas_Frecuente = mode(frutas_cercanas)
    return Mas_Frecuente