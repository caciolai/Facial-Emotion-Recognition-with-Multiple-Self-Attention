from setuptools import find_packages
from setuptools import setup

setup(
    name='caciolai-FER',
    version='1.0.0',
    description='Facial Emotion Recognition with Multiple Self-Attention.',
    author='caciolai',
    license='MIT License',
    url='https://github.com/caciolai/Facial-Emotion-Recognition-with-Multiple-Self-Attention',
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src
    python_requires='>=3.7.9',
    install_requires=[
        'tqdm'
        'matplotlib',
        'numpy',
        'opencv-python',
        'pandas',
        'Pillow',
        'scikit-learn',
        'seaborn',
        'torch',
        'pytorch-lightning',
        'fastai',
        'deepface',
    ]
)