---

## Universidad de Costa Rica
### Escuela de Ingeniería Eléctrica
#### IE0405 - Modelos Probabilísticos de Señales y Sistemas

---

* Estudiante 1: **Miguel Jiménez Tung**
* Carné 1: **B94104**
* Estudiante 2: **Alonso Jiménez Villegas**
* Carné 2: **B94125**
* Estudiante 3: **Daniel Pérez Conejo**
* Carné 3: **B85963**
* Grupo: **1 y 2**

---
# `P4` - *Modulación digital IQ*

### 4.1. - Modulación 8-PSK

* (50%) Realice una simulación del sistema de comunicaciones utilizando una modulación **8-PSK**. Deben mostrar las imágenes enviadas y recuperadas y las formas de onda.

Imagen recuperada con un SNR = 20:

![image](https://user-images.githubusercontent.com/93485961/143368429-e64b435f-0132-40cd-9cf8-46cbe9aa0688.png)

Imagen recuperada con un SNR = 5:

![image](https://user-images.githubusercontent.com/93485961/143371429-1480cd02-9ab3-4e95-bea3-d9fdf4c7031f.png)

Formas de onda:

![image](https://user-images.githubusercontent.com/93485961/143368524-778ea68f-2467-4e0e-8031-c5df9a17ce8e.png)

### 4.2. - Estacionaridad y ergodicidad

* (30%) Realice pruebas de estacionaridad y ergodicidad a la señal modulada `senal_Tx` y obtenga conclusiones sobre estas.

![image](https://user-images.githubusercontent.com/93485961/143371688-a04118b8-561e-44ae-bf88-0ef8b022348f.png)

En la gráfica anterior se muestra que el promedio de realizaciones y el valor teórico presentan el mismo valor, por lo que se puede concluir que hay ergocidad en la señal modulada (senal_Tx).

### 4.3. - Densidad espectral de potencia

* (20%) Determine y grafique la densidad espectral de potencia para la señal modulada `senal_Tx`.

Como se puede observar en la gráfica anterior, el punto más alto corresponde a una frecuencia de 5000Hz, lo que concuerda con el valor establecido para la frecuencia de la señal portadora. El resto de frecuencias se encuentran alrededor de este valor máximo, como era de esperar.
