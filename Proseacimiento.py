"""Necesito de aca poder obtener aquellas imagenes que sirvan
 
 - Contorno
 - Im Gris 
 - Im Binaria ---> Mascara
"""
import cv2
import numpy as np


def Aplicar_Mascara(img , mask):
    res = cv2.bitwise_and(img,img,mask = mask)
    cv2.imshow('Enmascaramiento',res)
    return res

def Apila_horizontal(imagenes):
    np_horizontal =  np.hstack(imagenes)
    # cv2.imshow(nombre,np_horizontal)
    return np_horizontal

def Mostrar_con_bordes(nombre,imagenes):
    
    bordes =list()
    mascaras = list()

    for im in imagenes :
        
        canny_temp = cv2.Canny(im,1,400)
        canny_1 = cv2.cvtColor(canny_temp, cv2.COLOR_GRAY2BGR)
        
        

        bordes.append(canny_1)
        
        _ ,mascara_rgb = Get_Mascara(im)
        mascaras.append(mascara_rgb)
        # Contornear(im)
    np_vert = np.vstack((Apila_horizontal(imagenes), Apila_horizontal(bordes),Apila_horizontal(mascaras)))
    

    cv2.imshow(nombre, np_vert)
    

def Get_Mascara(img):

    kernel = np.ones((5,5), np.uint8) 
    gris_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny_temp = cv2.Canny(gris_1,1,150)
    
    
    img_contours, _  = cv2.findContours(gris_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # thresh = cv2.adaptiveThreshold(gris_1,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,0)

    # cv2.drawContours(gris_1, img_contours, -1, (0, 0, 0))
    binaria = cv2.adaptiveThreshold(gris_1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,401,0)
    #retval, binaria = cv2.threshold(gris_1,0, 255 ,0)

    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    rgb = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
    return binaria, rgb



def Contornear(img):
    kernel = np.ones((5,5), np.uint8) 
    gris_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny_temp = cv2.Canny(gris_1,1,200)
    thresh = cv2.adaptiveThreshold(canny_temp,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,0)
    img_contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, img_contours, -1, (0, 255, 0))
    return img

def Get_Contorno(img):
    kernel_close = np.ones((9,9), np.uint8) 
    kernel_erosion = np.ones((3,3), np.uint8) 
    
    contorno = None
    gris = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gris,(5,5),0)
    # t, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)    
    th = cv2.adaptiveThreshold(gris, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV, 555 ,2)
    # ret, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    mascara = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)
    img_erosion = cv2.erode(mascara, kernel_erosion, iterations=1) 

    contours, _ = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        
        area = cv2.contourArea(c)

        if  (img.shape[0] * img.shape[1])*0.1 < area  and area < (img.shape[0] * img.shape[1])*0.75:
        
            contorno = c          
        # cv2.drawContours(img, [c], 0, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('imaima',img)

    

    return img_erosion, contorno


def Filtrados(imagen):
    """
    Recibe un IMAGEN\n
    DEVUELVE un MASCARA Y CONTONRNO 
    """


    
 # ---- PREPROCESAMIENTO Y FILTRADO ----------
    
   
    img = imagen.copy()
    
   
    
   
    # img_erosion = cv2.erode(img, kernel, iterations=1) 
    # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1) 
    kernel_close = np.ones((9,9), np.uint8) 
    kernel_erosion = np.ones((3,3), np.uint8) 
    contorno = None
  
    # PRUEBA 1
    
    # edge_1 = cv2.edgePreservingFilter( img, flags = 2, sigma_s= 50 , sigma_r = 0.30 )
    # style_0 = cv2.stylization(edge_1 , sigma_s = 60, sigma_r = 0.6 )
    edge = cv2.edgePreservingFilter( img, flags = 2, sigma_s= 50 , sigma_r = 0.50 )
    style = cv2.stylization(edge , sigma_s = 50, sigma_r = 0.4 )
    gris = cv2.cvtColor( style, cv2.COLOR_BGR2GRAY)
   
   
    
   
    # blur = cv2.GaussianBlur(gris,(5,5),0)
    # t, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)    
    th = cv2.adaptiveThreshold(gris, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV, 555 ,2)
    # ret, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    mascara = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)
    img_erosion = cv2.erode(mascara, kernel_erosion, iterations=1) 

    contours, _ = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    buenos_cnt =list()
    for c in contours:
        
        area = cv2.contourArea(c)

        if  (img.shape[0] * img.shape[1])*0.1 < area  and area < (img.shape[0] * img.shape[1])*0.75:
        
            buenos_cnt.append(c)          
        # cv2.drawContours(img, [c], 0, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('imaima',img)

    cnt = max(buenos_cnt, key=cv2.contourArea)

    # cv2.imshow('img orig', imagen)
    # cv2.imshow('edge', edge)    
    # cv2.imshow('style', style)
    # cv2.imshow('img_erosion', img_erosion)
    # Aplicar_Mascara(imagen,img_erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    MASCARA  = img_erosion    
    CONTORNO = cnt
    
    
    return MASCARA, CONTORNO
    

def PreProcesamiento(img):
   
    img_original = img.copy()
    edge = cv2.edgePreservingFilter( img_original, flags = 2, sigma_s= 50 , sigma_r = 0.50 )
    style = cv2.stylization(edge , sigma_s = 50, sigma_r = 0.4 )
    gris = cv2.cvtColor(style, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gris, (7,7), 0)

    canny = cv2.Canny(gauss, 50, 150)
    canny = cv2.dilate(canny , None, iterations= 2)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, (15,5))
    canny = cv2.erode(canny, (9,9), iterations=1)
  
    (contornos,_) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buenos_contornos = list()
    for c in contornos:
            
            area = cv2.contourArea(c)

            if  (img_original.shape[0] * img_original.shape[1])*0.3 < area  and area < (img_original.shape[0] * img_original.shape[1])*0.75:
            
                buenos_contornos.append( c )  
    cnt = max(contornos, key=cv2.contourArea)
        
    aux =  img_original.copy()
    cv2.drawContours(aux,[cnt],-1,(255,255,255), -1)
    gris = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # th = cv2.adaptiveThreshold(gauss, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, 555 ,2)
    kernel_close = np.ones((7,7), np.uint8) 
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,kernel_close)
    gauss_th = cv2.GaussianBlur(th, (15,15), 0)

    canny_th = cv2.Canny(gauss_th, 50, 150)
    canny_th= cv2.dilate(canny_th , None, iterations= 1)

    contours, _ = cv2.findContours(canny_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    contorno_grandes =list()
    for c in contours:
        
        area = cv2.contourArea(c)
        
        
        if  (img_original.shape[0] * img_original.shape[1])*0.1 < area  and area < (img_original.shape[0] * img_original.shape[1])*0.75:
    
            print('entro')
            contorno_grandes.append(c)         
    contorno_final  = np.concatenate(contorno_grandes, axis=0)
    cnt_1 = max(contorno_grandes, key=cv2.contourArea)

    th = cv2.erode(th, (9,9), iterations=1)
    # cv2.imshow('edge', style)
    # cv2.drawContours(img_original,[cnt],-1,(0,0,255), 1)

    # cv2.imshow('img original',img_original)
    # cv2.imshow('canny',canny_th)

    # res = cv2.bitwise_and(img_original,img_original ,mask = th)
    # cv2.imshow('Enmascaramiento',res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return th, cnt_1


if __name__ == "__main__":
    imagen = cv2.imread("img/banana_5.jpg")
    imagen = cv2.resize(imagen,(300,300))
    mascara , contorno = Filtrados(imagen)
    x, y, width, height = cv2.boundingRect(contorno)
    # cv2.imshow('Im original', mascara)
    roi = cv2.resize(mascara[y:y + height, x:x + width], (50, 50))
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    print(moments)
    cv2.imshow('Region interes', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()