from setuptools import setup, find_packages

setup(
    name='MateGen',  
    version='0.1.79',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'MateGen': ['.env'],
    },    
    install_requires=[
        'IPython',
        'openai>=1.35',
        'matplotlib',
        'pandas',
        'seaborn',
        'oss2',
        'python-dotenv',
        'pymysql',
        'requests',
        'google-api-python-client',
        'google-auth',
        'google-auth-oauthlib',
        'beautifulsoup4',
        'python-dateutil',
        'tiktoken',
        'lxml',
        'cryptography',
        'numpy',
        'html2text',
        'nbconvert'
    ],
    entry_points={
        'console_scripts': [
            'mategen=MateGen.mateGenClass:main',
        ],
    },
    author='Jiutian',
    author_email='2323365771@qq.com',
    description='交互式智能编程助手MateGen',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)