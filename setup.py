import setuptools

setuptools.setup(
    name='yowo',
    version='0.0.1',
    packages=['yowo'],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'Pillow',
        'moviepy',
        'sklearn',
        'torch',
        'pytorch-lightning'
    ],
)
