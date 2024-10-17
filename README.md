```mermaid
graph TD;
    HyperShare;
    HyperShare-->isomatrixpython;
    HyperShare-->isomatrixpython-->src;
    HyperShare-->isomatrixpython-->include;
    HyperShare-->isomatrixpython-->scripts;
    HyperShare-->isomatrixpython-->doc;
    HyperShare-->isomatrixpython-->lib;
    HyperShare-->isomatrixpython-->pred;

    

```



![accuracy](https://github.com/provallo-com/Hypershare/blob/main/AccuracyPerformance.png?raw=true)

build:

mkdir build && cd build &&  cmake ../
make -j8 all

