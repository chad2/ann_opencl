@title[Artificial neural network - CPU & OpenCL]
[https://www.github.com/chad2/annopencl]
---

@title[Neuronale Netze]
---

@title[Anwendung]
![](http://pytorch.org/static/img/horse2zebra.gif)
http://pytorch.org/static/img/horse2zebra.gif
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
---

@title[Funktionsweise]
http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#shallow-neural-network-graph
---

@title[Forwardpass]
http://cs231n.github.io/assets/nn1/neuron_model.jpeg
a(input * w + b) = y
init w,b
resulting error    cross entropy loss
---

@title[Backprop]
calc gradients - partielle ableitungen dL/dparam
matmul example
bias example
act example (relu)
kettenregel example
---

@title[Gradient descent]
step parameter w,b into negative gradient direction
learning rate + decay
<svg width="420" height="220"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="white" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="white">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="white" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="white">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="white" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="white">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="white" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="white">q</text><circle cx="170" cy="65" fill="white" stroke="white" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="white" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="white">f</text><circle cx="340" cy="117" fill="white" stroke="white" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="white" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>

http://cs231n.github.io/optimization-2/
---

@title[Begrifflichkeiten]
# init
# Epochen
# batches

---

@title[problemstellung]
bildklassifizierung
![](http://neuralnetworksanddeeplearning.com/images/ensemble_errors.png)
#### correct top
#### class bottom
---

@title[Praktikumsaufgabe]
cpu / opencl
---

@title[Codevorstellung]
uml here



---
<!--
    titel

github link

neuronale netze
    - anwendungen / Bestandteilepiele(bilder)
    - funktionsweise
        - Bestandteile(https://jalammar.github.io/visual-interactive-guide-basics-neural-networks#classification)
        - Forwardpass
        - backprop
        - update
    - epochs/lr-> w.b, decay /  / batches / loss / acc / init / activation
    - 

problemstellung
    - bildklassifizierung
    - mnist examples

praktikumsaufgabe
    - cpu / opencl 

codevorstellung
    - uml 

/////////////////
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