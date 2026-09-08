# 路由服务只用标准库，所以运行镜像不装任何第三方包。
# numpy / scikit-learn 只有跑 bench 与训练预测器时才需要（见 requirements-dev.txt）。
FROM python:3.11-slim

WORKDIR /app
COPY campus_bus/ ./campus_bus/
COPY network/ ./network/

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080

# 镜像内不落 numpy：Network.load 里的 numpy 导入只在解析速度矩阵时用到，
# 这里用纯 Python 版的加载路径（see campus_bus/network.py::Network.load）。
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).status==200 else 1)"

CMD ["python", "-m", "campus_bus.service", "--port", "8080", "--host", "0.0.0.0"]
