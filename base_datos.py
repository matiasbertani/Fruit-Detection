import pandas as pd
from Fruta import Fruta
class Base_Datos:

    def __init__(self, list_fruits) -> None:
        
        # Lista de todas las frutas
        self.FRUTAS = list_fruits  #TENDRIA QUE SER UN DICCIONARIO
        
        
        self.Dic_Frutas = dict()
        for fruta in list_fruits:
            self.Dic_Frutas[fruta.Nombre_Archivo] = fruta
        
        self.vector_minimos = list()
        self.vector_maximos = list()

        self.Calcular_MinMax()

        self.clusters= list()  #LISTA  con los nombres de los grupos

    def Calcular_MinMax(self):
        
        #estas tienen que ser lista llenas de cero del tama;ano del vector de caracteristicas
        list_min_aux = list()
        list_max_aux = list()

        
        for i in range( len( self.FRUTAS[0].VECTOR_CARACTERISTICAS ) ):
            
            aux_caracteristicas = list()            
            
            for fruta in self.FRUTAS:
                aux_caracteristicas.append(fruta.VECTOR_CARACTERISTICAS[i])

            list_min_aux.append( min( aux_caracteristicas ) )
            list_max_aux.append( max( aux_caracteristicas ) )

        self.vector_minimos = list_min_aux
        self.vector_maximos = list_max_aux

    def Armar_Matriz_Caracterristicas(self):
        """Genera un Dataframe con todas los vecrtores de cada fruta"""
        
        columnas = list()
        columnas.append('NOMBRE ARCHIVO')
        for i in range(len( self.FRUTAS[0].VECTOR_CARACTERISTICAS )):
            
            columnas.append(f'PROP {i}')
        
        mtr_caract = pd.DataFrame(columns = columnas)
        
        
        for fruta in self.FRUTAS:
            vec = list()
            vec.append( fruta.Nombre_Archivo )
            vec.extend( fruta.VECTOR_CARACTERISTICAS )        
            mtr_caract.loc[  len( mtr_caract )  ] = vec
            del(vec)

        mtr_caract.to_csv('../CACRACTERISTICAS.csv',sep=';',encoding='ANSI',index=False)
        self.MATRIZ_CARACTERISTICA = mtr_caract

    def Normalizar_Datos(self):
        
        resultado = self.MATRIZ_CARACTERISTICA.copy()
        
        for col_name in resultado.columns:
            if col_name != 'NOMBRE ARCHIVO':
                max_value = self.MATRIZ_CARACTERISTICA[col_name].max()
                min_value = self.MATRIZ_CARACTERISTICA[col_name].min()

                resultado[col_name] = (self.MATRIZ_CARACTERISTICA[col_name] - min_value) / (max_value - min_value)
        
        
        resultado.to_csv('../normalizados.csv',sep=';',encoding='ANSI',index=False)
        resultado.set_index('NOMBRE ARCHIVO', inplace = True)
        
        for id in resultado.index:
            self.Dic_Frutas[ id ].NORM_VECTOR_CARACTERISTICAS = tuple(  resultado.loc[id]  )

        

    def Get_df_knn( self,fruta_knn ) -> pd.DataFrame:
        "recibe una fruta, y calcula las distanciascon cada fruuta en la lista y arma un df"
        
        
        registro_df = list()

        for fruit in self.FRUTAS:
            distancia = fruit.Cacular_Distancia_Vector(fruta_knn)
            registro_df.append( [ fruit.NOMBRE, distancia ] )

        df_resultado = pd.DataFrame(  registro_df,
                               columns= ['NOMBRE', 'DISTANCIA'] )
        return df_resultado
