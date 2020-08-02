import setuptools


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':

    setuptools.setup(
        name="tjevents",
        version="0.1",
        author="Kai Chen",
        author_email="14sdck@tongji.edu.cn",
        description="A general deep learning based event camera data processing toolbox "
                    "from Institute of Intelligent Vehicle at Tongji University",
        long_description=readme(),
        packages=setuptools.find_packages(exclude=('config', 'tools')),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )
