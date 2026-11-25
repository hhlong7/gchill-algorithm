FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gfortran cmake git ca-certificates pkg-config python3 \
      libopenblas-dev liblapacke-dev libstarpu-dev starpu-tools libfxt-dev libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.4.0 --recursive https://gitlab.inria.fr/solverstack/chameleon.git /src/chameleon
RUN cmake -S /src/chameleon -B /src/chameleon/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/chameleon -DCHAMELEON_USE_MPI=OFF -DCHAMELEON_USE_CUDA=OFF \
      -DCHAMELEON_PREC_C=OFF -DCHAMELEON_PREC_Z=OFF -DCHAMELEON_ENABLE_EXAMPLE=OFF \
      -DCHAMELEON_ENABLE_TESTING=ON -DBLA_VENDOR=OpenBLAS \
    && cmake --build /src/chameleon/build -j 6 && cmake --install /src/chameleon/build
ENV PATH=/opt/chameleon/bin:$PATH
WORKDIR /work