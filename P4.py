#Proyecto 4

#Estudiante: Miguel Jiménez, Alonso Jiménez y Daniel Pérez
#Carné: B94104, B94125 y B85963
#Grupo: 1

#Antes de comenzar con el código de las asignaciones, se necesitan las funciones definidas en la parte 3 del documento

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
 
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bits):
        if bit == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora
            moduladora[i*mpp : (i+1)*mpp] = 1
        else:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora * -1
            moduladora[i*mpp : (i+1)*mpp] = 0
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora, moduladora

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

def demodulador(senal_Rx, portadora, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora * portadora)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadora
        Ep = np.sum(producto) 
        senal_demodulada[i*mpp : (i+1)*mpp] = producto

        # Criterio de decisión por detección de energía
        if Ep > 0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

#Para el proyecto
#4.1

def modulador8PSK(bits, fc, mpp):
    '''Un método que simula el esquema de modulación digital 8-PSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La primera onda portadora c(t)
    :return: La segunda onda portadora c(t)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp) # mpp: muestras por periodo
    portadora_1 = np.cos(2*np.pi*fc*t_periodo) # Señal portadora con seno
    portadora_2 = np.sin(2*np.pi*fc*t_periodo) # Señal portadora con coseno
    
    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)

    # 4. Asignar las formas de onda según los bits (8-PSK)
    h = np.sqrt(2)/2 # De primero se define el valor de h
    
    # Para los bits b1 b2 b3, se tienen las siguientes combinaciones:
    for i in range(0, N, 3):
        
    # Para b1b2b3 = 111, A1 = 1 y A2 = 0
        if (bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 1):
            senal_Tx[i*mpp: (i+1)*mpp] = 1*portadora_1 + 0*portadora_2
            
    # Para b1b2b3 = 110, A1 = h y A2 = h
        elif (bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 0):
            senal_Tx[i*mpp: (i+1)*mpp] = h*portadora_1 + h*portadora_2
            
    # Para b1b2b3 = 010, A1 = 0 y A2 = 1
        elif (bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 0):
            senal_Tx[i*mpp: (i+1)*mpp] = 0*portadora_1 + 1*portadora_2
            
    # Para b1b2b3 = 011, A1 = -h y A2 = h
        elif (bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 1):
            senal_Tx[i*mpp: (i+1)*mpp] = -h*portadora_1 + h*portadora_2
            
    # Para b1b2b3 = 001, A1 = -1 y A2 = 0
        elif (bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 1):
            senal_Tx[i*mpp: (i+1)*mpp] = -1*portadora_1 + 0*portadora_2
            
    # Para b1b2b3 = 000, A1 = -h y A2 = -h
        elif (bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 0):
            senal_Tx[i*mpp: (i+1)*mpp] = -h*portadora_1 + -h*portadora_2
            
    # Para b1b2b3 = 100, A1 = 0 y A2 = -1
        elif (bits[i] == 1 and bits[i+1] == 0 and bits[i+2] == 0):
            senal_Tx[i*mpp: (i+1)*mpp] = 0*portadora_1 + -1*portadora_2
            
    # Para b1b2b3 = 101, A1 = h y A2 = -h
        else:
            senal_Tx[i*mpp: (i+1)*mpp] = h*portadora_1 + -h*portadora_2
        
# 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = 1 / (N*Tc) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora_1, portadora_2

def demodulador8PSK(senal_Rx, portadora_1, portadora_2, mpp):
    '''Un método que simula un bloque demodulador 
    de señales, bajo un esquema 8-PSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    :return: Primera señal demodulada
    :return: Segunda señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)
    
    # Valor de h
    h = np.sqrt(2)/2

    # Vector para la señal demodulada
    demodulada_1 = np.zeros(senal_Rx.shape)
    demodulada_2 = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de las señales portadoras
    E1 = np.sum(portadora_1 * portadora_1) # Portadora 1 
    E2 = np.sum(portadora_2 * portadora_2) # Portadora 2 
    
    # Demodulación
    for i in range(N):
        #Producto interno de dos funciones
        producto_1 = senal_Rx[i*mpp : (i+1)*mpp] * portadora_1
        producto_2 = senal_Rx[i*mpp : (i+1)*mpp] * portadora_2
        
        # Umbrales de energía
        A1 = np.sum(producto_1) # Portadora 1
        A2 = np.sum(producto_2) # Portadora 2
        
        # Señales demoduladas
        demodulada_1[i*mpp : (i+1)*mpp] = producto_1
        demodulada_2[i*mpp : (i+1)*mpp] = producto_2
    
        #Se utilizan como criterios:
            #E < -(1+h)/2 ----> -1
            #-h/2 > E > -(1+h)/2 ----> -h
            #-h/2 < E < h/2 ----> 0
            #h/2 < E < (1+h)/2 ----> h
            #E > (1+h)/2 ----> 1
    
        if i % 3 == 0:  
        # Para b1b2b3 = 111, A1 = 1 y A2 = 0
            if (A1 >= (1+h)*E1/2 and A2 >= -h*E2/2 and A2 <= h*E2/2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
                
        # Para b1b2b3 = 110, A1 = h y A2 = h
            elif (A1 >= h*E1/2 and A1 <= (1+h)*E1/2 and A2 >= h*E2/2 and A2 <= (1+h)*E2/2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
                
        # Para b1b2b3 = 010, A1 = 0 y A2 = 1
            elif (A1 >= -h*E1/2 and A1 <= h*E1/2 and A2 >= (1+h)*E2/2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
                
        # Para b1b2b3 = 011, A1 = -h y A2 = h
            elif (A1 >= -(h+1)*E1/2 and A1 <= -h*E1/2 and A2 >= h*E2/2 and A2 <= (1+h)*E2/2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
                
        # Para b1b2b3 = 001, A1 = -1 y A2 = 0
            elif (A1 <= -(1+h)*E1/2 and A2 >= -h*E2/2 and A2 <= h*E2/2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
                
        # Para b1b2b3 = 000, A1 = -h y A2 = -h
            elif (A1 >= -(h+1)*E1/2 and A1 <= -h*E1/2 and A2>= -(h+1)*E2/2 and A2 <= -h*E2/2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
                
        # Para b1b2b3 = 100, A1 = 0 y A2 = -1
            elif (A1 >= -h*E1/2 and A1 <= h*E1/2 and A2 <= -(1+h)*E2/2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
                
        # Para b1b2b3 = 101, A1 = h y A2 = -h
            else:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
                
    return bits_Rx.astype(int), demodulada_1, demodulada_2

fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 20   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema 8-PSK
senal_Tx, Pm, portadora_1, portadora_2 = modulador8PSK(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, demodulada_1, demodulada_2 = demodulador8PSK(senal_Rx, portadora_1, portadora_2, mpp)

# 6. Se visualiza la imagen recibida
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10, 6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

#4.2

# Frecuencia de Portadoras
fc = 5000

# Número de elementos
T = 100  
# Tiempo final
tf = 10    
# Vector de tiempo 
t = np.linspace(0, tf, T)

# Matriz de muestras
n = 8 # Se tienen 8 posibilidades de símbolos
Stx = np.empty((n, len(t))) 

# Se crean listas con las variables aleatorias
h = np.sqrt(2)/ 2

A1 = [1, h, 0, -h, -1, -h, 0, h]
A2 = [0, h, 1, h, 0, -h, -1, -h]

# Se crea un for para evaluar las posibilidades de las muestras
for i in range(0,len(A1)):
    St = A1[i]*np.cos(2*np.pi*fc*t) + A2[i]*np.sin(2*np.pi*fc*t)
    Stx[i,:] = St
    plt.plot(t, St)
    
# Promedio de las muestras
Prom = [np.mean(Stx[:,i]) for i in range(len(t))]
Muestras = plt.plot(t, Prom, lw=4, color = 'grey', label='Promedio muestras')

# Promedio de la senal_Tx
Vstx = np.mean(senal_Tx)*t
Teorico = plt.plot(t, Vstx, '--', lw=4, color = 'k', label='senal_Tx')

plt.title('Realizaciones del proceso aleatorio $s(t)$')
plt.xlabel('$t$')
plt.ylabel('$s(t)$')
plt.legend()
plt.show() 

#4.3

from scipy import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()
