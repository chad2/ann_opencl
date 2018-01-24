@title[Artificial neural network - CPU & OpenCL]
## Artificial neural network - C++ & OpenCL
[https://www.github.com/chad2/annopencl]
---

@title[Neuronale Netze]
## Neuronale Netze
---

@title[Anwendung]
### Anwendung

##### Texterkennung
##### Bilderkennung
##### Gesichtserkennung

![](https://github.com/tensorflow/models/raw/master/research/object_detection/g3doc/img/kites_detections_output.jpg&size=auto)

Note:
- objekterkennung - autonomes fahren
- hier: simple klassifizierung
---

@title[Funktionsweise]
### Funktionsweise

![](https://cs231n.github.io/assets/nn1/neural_net.jpeg)

Note:
- input layer dim 784 (28*28)
- output layer dim 10
- 3x4 weights + 4 biases
- 4x2 weights + 2 biases
- act between

---
@title[Forwardpass]
### Forwardpass

![](http://cs231n.github.io/assets/nn1/neuron_model.jpeg)

Note:
- random init b/w
---

@title[Aktivierungsfunktion]
### Aktivierungfunktion

![](https://cs231n.github.io/assets/nn1/relu.jpeg)
#### Relu

![](https://cs231n.github.io/assets/nn1/tanh.jpeg)
#### Hyperbolic Tangent

Note:
- ohne act nur lineare klassifizierung
- 2d linie / 3d ebene
- muss differenzierbar sein, drelu = 0

---

@title[Probabilities + Loss]
### Probabilities + Loss

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mstyle displaystyle="true" scriptlevel="0">
    <mrow class="MJX-TeXAtom-ORD">
      <msub>
        <mi>L</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>=</mo>
      <mo>&#x2212;<!-- − --></mo>
      <mi>log</mi>
      <mrow>
        <mo>(</mo>
        <mfrac>
          <msup>
            <mi>e</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <msub>
                <mi>f</mi>
                <mrow class="MJX-TeXAtom-ORD">
                  <msub>
                    <mi>y</mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mi>i</mi>
                    </mrow>
                  </msub>
                </mrow>
              </msub>
            </mrow>
          </msup>
          <mrow>
            <munder>
              <mo>&#x2211;<!-- ∑ --></mo>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>j</mi>
              </mrow>
            </munder>
            <msup>
              <mi>e</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <msub>
                  <mi>f</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mi>j</mi>
                  </mrow>
                </msub>
              </mrow>
            </msup>
          </mrow>
        </mfrac>
        <mo>)</mo>
      </mrow>
    </mrow>
  </mstyle>
</math>

#### TODO - vector example

Note:
- i index of correct label/class
- loss needs to reduce
---

@title[Backprop]
### Partiell nach allen Parametern ableiten

<svg width="420" height="220"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="white" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="white">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="white" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="white">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="white" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="white">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="white" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="white">q</text><circle cx="170" cy="65" fill="white" stroke="white" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="white" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="white">f</text><circle cx="340" cy="117" fill="white" stroke="white" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>

#### TODO - 2 Ableitungen

Note:
- Gradienten für Inputs unnötig
---

@title[Gradienten Abstiegsverfahren]
### Gradienten Abstiegsverfahren
$$p = p + \frac{\partial L}{\partial p} (-\lambda)$$
![](https://cs231n.github.io/assets/nn3/opt2.gif)

Note:
- use gradient from backprop
- step parameter w,b into negative gradient direction
- learning rate + decay
---

@title[Begrifflichkeiten]
### Begrifflichkeiten
#### Epoche
#### Batch
###### TODO - Messung 1
###### TODO - Messung 2

Note:
- Epoche: Ein Durchlauf durch alle Trainingsdaten
- Batch: Vektor-Multiplikation zu Matrix-Vektor-Multiplikation

---

@title[Praktikumsaufgabe]
### Praktikumsaufgabe

#### Hanschriftenerkennung MNIST Datensatz
#### Implementierung:
1. CPU mit C++
2. CPU mit C++ und MatrixMultiplikation mit GPU (OpenCL)
3. Komplettes Netz auf GPU

---

@title[Problemstellung]
### Problemstellung
![](http://neuralnetworksanddeeplearning.com/images/ensemble_errors.png)

#### 60.000 Trainingsdate & 10.000 Testdaten
#### Überwachtes Lernen

Note:
- bildklassifizierung
- korrekt top
- classification bottom
---

@title[Codevorstellung]
### Codevorstellung
<svg fill-opacity="1" xmlns:xlink="http://www.w3.org/1999/xlink" color-rendering="auto" color-interpolation="auto" text-rendering="auto" stroke="white" stroke-linecap="square" width="710" stroke-miterlimit="10" shape-rendering="auto" stroke-opacity="1" fill="white" stroke-dasharray="none" font-weight="normal" stroke-width="1" viewBox="350 310 710 270" height="270" xmlns="http://www.w3.org/2000/svg" font-family="'Dialog'" font-style="normal" stroke-linejoin="miter" font-size="12px" stroke-dashoffset="0" image-rendering="auto"
><!--Generated by the Batik Graphics2D SVG Generator--><defs id="genericDefs"
  /><g
  ><defs id="defs1"
    ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath1"
      ><path d="M0 0 L2147483647 0 L2147483647 2147483647 L0 2147483647 L0 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath2"
      ><path d="M0 0 L0 30 L100 30 L100 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath3"
      ><path d="M0 0 L0 30 L160 30 L160 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath4"
      ><path d="M0 0 L0 140 L300 140 L300 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath5"
      ><path d="M0 0 L0 110 L330 110 L330 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath6"
      ><path d="M0 0 L0 100 L330 100 L330 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath7"
      ><path d="M0 0 L0 40 L130 40 L130 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath8"
      ><path d="M0 0 L0 40 L80 40 L80 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath9"
      ><path d="M0 0 L0 30 L80 30 L80 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath10"
      ><path d="M0 0 L0 110 L210 110 L210 0 Z"
      /></clipPath
      ><clipPath clipPathUnits="userSpaceOnUse" id="clipPath11"
      ><path d="M0 0 L0 30 L140 30 L140 0 Z"
      /></clipPath
    ></defs
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(450,440)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(450,440)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="11" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >imageLabel</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(450,490)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(450,490)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="26" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >Reader</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(880,440)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="158.5" height="28.5" y="0.5" clip-path="url(#clipPath3)" stroke="none"
    /></g
    ><g transform="translate(880,440)"
    ><rect fill="none" x="0.5" width="158.5" height="28.5" y="0.5" clip-path="url(#clipPath3)"
      /><text x="7" font-size="14px" y="17.8281" clip-path="url(#clipPath3)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >AnnOpenCL_kernel.cl</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(680,440)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(680,440)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="9" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >AnnOpenCL</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(910,360)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(910,360)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="21" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >kernel.cl</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(760,360)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(760,360)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="31" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >clMul</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(400,360)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(400,360)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="33" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >main</text
    ></g
    ><g fill="rgb(255,255,255)" fill-opacity="0" transform="translate(610,360)" stroke-opacity="0" stroke="rgb(255,255,255)"
    ><rect x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)" stroke="none"
    /></g
    ><g transform="translate(610,360)"
    ><rect fill="none" x="0.5" width="98.5" height="28.5" y="0.5" clip-path="url(#clipPath2)"
      /><text x="36" font-size="14px" y="17.8281" clip-path="url(#clipPath2)" font-family="sans-serif" stroke="none" xml:space="preserve"
      >Ann</text
    ></g
    ><g stroke-dasharray="8,5" stroke-miterlimit="5" transform="translate(370,330)" stroke-linecap="butt"
    ><path fill="none" d="M79.5 120.5 L10.5 120.5" clip-path="url(#clipPath4)"
      /><path fill="none" d="M10.5 120.5 L10.5 10.5" clip-path="url(#clipPath4)"
      /><path fill="none" d="M10.5 10.5 L280.5 10.5" clip-path="url(#clipPath4)"
      /><path fill="none" d="M280.5 10.5 L280.5 30.5" clip-path="url(#clipPath4)"
      /><path fill="white" d="M68.7417 114 L80 120.5 L68.7417 127 Z" clip-path="url(#clipPath4)" stroke="none"
      /><path fill="none" stroke-miterlimit="10" stroke-dasharray="none" d="M68.7417 114 L80 120.5 L68.7417 127 Z" clip-path="url(#clipPath4)" stroke-linecap="square"
    /></g
    ><g font-family="sans-serif" font-size="14px" transform="translate(370,330)"
    ><text x="14" xml:space="preserve" y="104.5703" clip-path="url(#clipPath4)" stroke="none"
      >«uses»</text
    ></g
    ><g stroke-dasharray="8,5" stroke-miterlimit="5" transform="translate(420,440)" stroke-linecap="butt"
    ><path fill="none" d="M29.5 10.5 L10.5 10.5" clip-path="url(#clipPath5)"
      /><path fill="none" d="M10.5 10.5 L10.5 90.5" clip-path="url(#clipPath5)"
      /><path fill="none" d="M10.5 90.5 L310.5 90.5" clip-path="url(#clipPath5)"
      /><path fill="none" d="M310.5 90.5 L310.5 30.5" clip-path="url(#clipPath5)"
      /><path fill="white" d="M18.7417 4 L30 10.5 L18.7417 17 Z" clip-path="url(#clipPath5)" stroke="none"
      /><path fill="none" stroke-miterlimit="10" stroke-dasharray="none" d="M18.7417 4 L30 10.5 L18.7417 17 Z" clip-path="url(#clipPath5)" stroke-linecap="square"
    /></g
    ><g stroke-dasharray="8,5" stroke-miterlimit="5" transform="translate(420,460)" stroke-linecap="butt"
    ><path fill="none" d="M29.5 40.5 L10.5 40.5" clip-path="url(#clipPath6)"
      /><path fill="none" d="M10.5 40.5 L10.5 70.5" clip-path="url(#clipPath6)"
      /><path fill="none" d="M10.5 70.5 L310.5 70.5" clip-path="url(#clipPath6)"
      /><path fill="none" d="M310.5 70.5 L310.5 10.5" clip-path="url(#clipPath6)"
      /><path fill="white" d="M18.7417 34 L30 40.5 L18.7417 47 Z" clip-path="url(#clipPath6)" stroke="none"
      /><path fill="none" stroke-miterlimit="10" stroke-dasharray="none" d="M18.7417 34 L30 40.5 L18.7417 47 Z" clip-path="url(#clipPath6)" stroke-linecap="square"
    /></g
    ><g font-family="sans-serif" font-size="14px" transform="translate(420,460)"
    ><text x="14" xml:space="preserve" y="86.6562" clip-path="url(#clipPath6)" stroke="none"
      >«uses»</text
    ></g
    ><g stroke-dasharray="8,5" stroke-miterlimit="5" transform="translate(770,440)" stroke-linecap="butt"
    ><path fill="none" d="M109.5 20.5 L10.5 20.5" clip-path="url(#clipPath7)"
      /><path fill="white" d="M98.7417 14 L110 20.5 L98.7417 27 Z" clip-path="url(#clipPath7)" stroke="none"
      /><path fill="none" stroke-miterlimit="10" stroke-dasharray="none" d="M98.7417 14 L110 20.5 L98.7417 27 Z" clip-path="url(#clipPath7)" stroke-linecap="square"
    /></g
    ><g font-family="sans-serif" font-size="14px" transform="translate(770,440)"
    ><text x="38.2173" xml:space="preserve" y="16" clip-path="url(#clipPath7)" stroke="none"
      >«uses»</text
    ></g
    ><g stroke-dasharray="8,5" stroke-miterlimit="5" transform="translate(850,360)" stroke-linecap="butt"
    ><path fill="none" d="M59.5 20.5 L10.5 20.5" clip-path="url(#clipPath8)"
      /><path fill="white" d="M48.7417 14 L60 20.5 L48.7417 27 Z" clip-path="url(#clipPath8)" stroke="none"
      /><path fill="none" stroke-miterlimit="10" stroke-dasharray="none" d="M48.7417 14 L60 20.5 L48.7417 27 Z" clip-path="url(#clipPath8)" stroke-linecap="square"
    /></g
    ><g font-family="sans-serif" font-size="14px" transform="translate(850,360)"
    ><text x="13.2173" xml:space="preserve" y="16" clip-path="url(#clipPath8)" stroke="none"
      >«uses»</text
    ></g
    ><g transform="translate(700,370)"
    ><path fill="none" d="M59.5 10.5 L10.5 10.5" clip-path="url(#clipPath9)"
      /><path fill="white" d="M48.7417 4 L60 10.5 L48.7417 17 Z" clip-path="url(#clipPath9)" stroke="none"
      /><path fill="none" d="M48.7417 4 L60 10.5 L48.7417 17 Z" clip-path="url(#clipPath9)"
    /></g
    ><g transform="translate(490,370)"
    ><path fill="none" d="M189.5 90.5 L80.5 90.5" clip-path="url(#clipPath10)"
      /><path fill="none" d="M80.5 90.5 L80.5 10.5" clip-path="url(#clipPath10)"
      /><path fill="none" d="M80.5 10.5 L10.5 10.5" clip-path="url(#clipPath10)"
      /><path fill="white" d="M178.7417 84 L190 90.5 L178.7417 97 Z" clip-path="url(#clipPath10)" stroke="none"
      /><path fill="none" d="M178.7417 84 L190 90.5 L178.7417 97 Z" clip-path="url(#clipPath10)"
    /></g
    ><g transform="translate(490,370)"
    ><path fill="none" d="M119.5 10.5 L10.5 10.5" clip-path="url(#clipPath11)"
      /><path fill="white" d="M108.7417 4 L120 10.5 L108.7417 17 Z" clip-path="url(#clipPath11)" stroke="none"
      /><path fill="none" d="M108.7417 4 L120 10.5 L108.7417 17 Z" clip-path="url(#clipPath11)"
    /></g
  ></g
></svg>

---

@title[Referenzen]
### Referenzen
#### https://github.com/tensorflow/models/tree/master/research/object_detection
#### https://cs231n.github.io/
---

<!--
Code show:

main
    - makefile macros
    - const params
    - loop explain content
ann.cpp
    - init
    - forward
    - backprop
    - update
    - mulmat überleitung
    - padding

clmul.cpp
    - matmul
    - kernel

annopencl.cpp
    - speicheralloc
    - kernel parameter

annopencl_kernel.cl
    - forward
    - backprop
    - update

demo
    - cpu only
    - cpu + opencl matmutl
    - full opencl

-->
