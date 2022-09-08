from setuptools import find_packages, setup

setup(
    name="habitat_manipulation",
    author="Jiayuan Gu",
    packages=find_packages(
        include=[
            "habitat_extensions*",
            "mobile_manipulation*",
        ]
    ),
)
