from setuptools import setup


setup(
    name='my_custom_sklearn_transforms',
    version='1.0',
    description='''
            Custom transforms from scikit-learn into Watson Machine Learning
                ''',
    url='https://github.com/rodrigolima82/sklearn_transforms',
    author='Rodrigo Lima Oliveira',
    author_email='rodrigolima82@gmail.com',
    license='BSD',
    packages=[
        'my_custom_sklearn_transforms'
    ],
    zip_safe=False,
    install_requires=[
        'imbalanced-learn==0.4.3',
        'xgboost==1.2.0'
    ],
)
