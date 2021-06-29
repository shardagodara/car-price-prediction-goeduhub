{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.3"
    },
    "colab": {
      "name": "used_car_price_prediction.ipynb",
      "provenance": []
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DGu7Gj6G4xKP"
      },
      "source": [
        "# USED CAR PREDICTION PRICE"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AhLADDqk4xKQ"
      },
      "source": [
        "<img src=\"https://www.marketingdonut.co.uk/sites/default/files/styles/landing_pages_lists/public/usedcardealer1.jpg?itok=lSSEdwpY\">"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "exGf0iaJ4xKR"
      },
      "source": [
        "# Problem definition\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y2ZYuNPz4xKS"
      },
      "source": [
        "This is the first step of machine learning life cycle.Here we analyse what kind of problem is, how to solve it. \n",
        "So for this project we are using a car dataset, where we want to predict the selling price of car based on its certain features.\n",
        "Since we need to find the real value, with real calculation, therefore this problem is regression problem. \n",
        "We will be using regression machine learning algorithms to solve this problem."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OCyGYyhB4xKT"
      },
      "source": [
        "#loading required libraries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "%matplotlib inline"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kmHMbAN94xKU"
      },
      "source": [
        "# Data Gathering"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ncQ8cIPA4xKW"
      },
      "source": [
        "dataset = pd.read_csv('car_dataset.csv')"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0UMaBMtm4xKY"
      },
      "source": [
        "# Data Preparation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1F7r9iQn4xKZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9a4cfaee-d7ec-4d14-ee64-86acd22552ba"
      },
      "source": [
        "#checking no. of rows and columns in dataset\n",
        "dataset.shape"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(301, 9)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-yv7aXHT4xKa"
      },
      "source": [
        "This dataset contains 301 rows and 9 columns"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T7xlxj4f4xKb",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e5f2365c-e676-49ca-9c74-3ab99b078cab"
      },
      "source": [
        "#Checking the data type of columns.\n",
        "#this step is important because sometimes dataset may contain wrong datatype of the feature.\n",
        "dataset.info()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 301 entries, 0 to 300\n",
            "Data columns (total 9 columns):\n",
            " #   Column         Non-Null Count  Dtype  \n",
            "---  ------         --------------  -----  \n",
            " 0   Car_Name       301 non-null    object \n",
            " 1   Year           301 non-null    int64  \n",
            " 2   Selling_Price  301 non-null    float64\n",
            " 3   Present_Price  301 non-null    float64\n",
            " 4   Kms_Driven     301 non-null    int64  \n",
            " 5   Fuel_Type      301 non-null    object \n",
            " 6   Seller_Type    301 non-null    object \n",
            " 7   Transmission   301 non-null    object \n",
            " 8   Owner          301 non-null    int64  \n",
            "dtypes: float64(2), int64(3), object(4)\n",
            "memory usage: 21.3+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uV1CnO2i4xKc"
      },
      "source": [
        "Good! every data type is correctly mentioned. We need not to make any changes."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zqUX5Ti84xKc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 294
        },
        "outputId": "5e021eaa-d5cf-436b-cfb4-ebd2e5f7bd9f"
      },
      "source": [
        "#check statistical summary of all the columns with numerical values.\n",
        "dataset.describe()"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Year</th>\n",
              "      <th>Selling_Price</th>\n",
              "      <th>Present_Price</th>\n",
              "      <th>Kms_Driven</th>\n",
              "      <th>Owner</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>301.000000</td>\n",
              "      <td>301.000000</td>\n",
              "      <td>301.000000</td>\n",
              "      <td>301.000000</td>\n",
              "      <td>301.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>2013.627907</td>\n",
              "      <td>4.661296</td>\n",
              "      <td>7.628472</td>\n",
              "      <td>36947.205980</td>\n",
              "      <td>0.043189</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>2.891554</td>\n",
              "      <td>5.082812</td>\n",
              "      <td>8.644115</td>\n",
              "      <td>38886.883882</td>\n",
              "      <td>0.247915</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>2003.000000</td>\n",
              "      <td>0.100000</td>\n",
              "      <td>0.320000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>2012.000000</td>\n",
              "      <td>0.900000</td>\n",
              "      <td>1.200000</td>\n",
              "      <td>15000.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>2014.000000</td>\n",
              "      <td>3.600000</td>\n",
              "      <td>6.400000</td>\n",
              "      <td>32000.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>2016.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>9.900000</td>\n",
              "      <td>48767.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>2018.000000</td>\n",
              "      <td>35.000000</td>\n",
              "      <td>92.600000</td>\n",
              "      <td>500000.000000</td>\n",
              "      <td>3.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "              Year  Selling_Price  Present_Price     Kms_Driven       Owner\n",
              "count   301.000000     301.000000     301.000000     301.000000  301.000000\n",
              "mean   2013.627907       4.661296       7.628472   36947.205980    0.043189\n",
              "std       2.891554       5.082812       8.644115   38886.883882    0.247915\n",
              "min    2003.000000       0.100000       0.320000     500.000000    0.000000\n",
              "25%    2012.000000       0.900000       1.200000   15000.000000    0.000000\n",
              "50%    2014.000000       3.600000       6.400000   32000.000000    0.000000\n",
              "75%    2016.000000       6.000000       9.900000   48767.000000    0.000000\n",
              "max    2018.000000      35.000000      92.600000  500000.000000    3.000000"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Wu-MO5E_4xKd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "40fa36f9-4e61-4a7e-bb12-e2b4ac5680a7"
      },
      "source": [
        "#check if there is any missing value in the dataset\n",
        "dataset.isnull().sum()"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Car_Name         0\n",
              "Year             0\n",
              "Selling_Price    0\n",
              "Present_Price    0\n",
              "Kms_Driven       0\n",
              "Fuel_Type        0\n",
              "Seller_Type      0\n",
              "Transmission     0\n",
              "Owner            0\n",
              "dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "T02Um0sF4xKe"
      },
      "source": [
        "There are no missing values in the dataset"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gS9oBvsG4xKe"
      },
      "source": [
        "# Feature Engineering"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uFyJZTF24xKf",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 222
        },
        "outputId": "5dccbba5-808c-4cb7-a773-3185d1df6010"
      },
      "source": [
        "#adding a column with the current year\n",
        "dataset['Current_Year']=2020\n",
        "dataset.head(5)"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Car_Name</th>\n",
              "      <th>Year</th>\n",
              "      <th>Selling_Price</th>\n",
              "      <th>Present_Price</th>\n",
              "      <th>Kms_Driven</th>\n",
              "      <th>Fuel_Type</th>\n",
              "      <th>Seller_Type</th>\n",
              "      <th>Transmission</th>\n",
              "      <th>Owner</th>\n",
              "      <th>Current_Year</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ritz</td>\n",
              "      <td>2014</td>\n",
              "      <td>3.35</td>\n",
              "      <td>5.59</td>\n",
              "      <td>27000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>sx4</td>\n",
              "      <td>2013</td>\n",
              "      <td>4.75</td>\n",
              "      <td>9.54</td>\n",
              "      <td>43000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>ciaz</td>\n",
              "      <td>2017</td>\n",
              "      <td>7.25</td>\n",
              "      <td>9.85</td>\n",
              "      <td>6900</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>wagon r</td>\n",
              "      <td>2011</td>\n",
              "      <td>2.85</td>\n",
              "      <td>4.15</td>\n",
              "      <td>5200</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>swift</td>\n",
              "      <td>2014</td>\n",
              "      <td>4.60</td>\n",
              "      <td>6.87</td>\n",
              "      <td>42450</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Car_Name  Year  Selling_Price  ...  Transmission  Owner Current_Year\n",
              "0     ritz  2014           3.35  ...        Manual      0         2020\n",
              "1      sx4  2013           4.75  ...        Manual      0         2020\n",
              "2     ciaz  2017           7.25  ...        Manual      0         2020\n",
              "3  wagon r  2011           2.85  ...        Manual      0         2020\n",
              "4    swift  2014           4.60  ...        Manual      0         2020\n",
              "\n",
              "[5 rows x 10 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P1-swh3g4xKf",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 222
        },
        "outputId": "bc1a7c46-9ca9-4012-8f09-8d0c8bd46ba4"
      },
      "source": [
        "#creating a new column which will be age of vehicles; new feature\n",
        "dataset['Vehicle_Age']=dataset['Current_Year'] - dataset['Year']\n",
        "dataset.head(5)\n"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Car_Name</th>\n",
              "      <th>Year</th>\n",
              "      <th>Selling_Price</th>\n",
              "      <th>Present_Price</th>\n",
              "      <th>Kms_Driven</th>\n",
              "      <th>Fuel_Type</th>\n",
              "      <th>Seller_Type</th>\n",
              "      <th>Transmission</th>\n",
              "      <th>Owner</th>\n",
              "      <th>Current_Year</th>\n",
              "      <th>Vehicle_Age</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ritz</td>\n",
              "      <td>2014</td>\n",
              "      <td>3.35</td>\n",
              "      <td>5.59</td>\n",
              "      <td>27000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>sx4</td>\n",
              "      <td>2013</td>\n",
              "      <td>4.75</td>\n",
              "      <td>9.54</td>\n",
              "      <td>43000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>ciaz</td>\n",
              "      <td>2017</td>\n",
              "      <td>7.25</td>\n",
              "      <td>9.85</td>\n",
              "      <td>6900</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>wagon r</td>\n",
              "      <td>2011</td>\n",
              "      <td>2.85</td>\n",
              "      <td>4.15</td>\n",
              "      <td>5200</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>swift</td>\n",
              "      <td>2014</td>\n",
              "      <td>4.60</td>\n",
              "      <td>6.87</td>\n",
              "      <td>42450</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>0</td>\n",
              "      <td>2020</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Car_Name  Year  Selling_Price  ...  Owner  Current_Year Vehicle_Age\n",
              "0     ritz  2014           3.35  ...      0          2020           6\n",
              "1      sx4  2013           4.75  ...      0          2020           7\n",
              "2     ciaz  2017           7.25  ...      0          2020           3\n",
              "3  wagon r  2011           2.85  ...      0          2020           9\n",
              "4    swift  2014           4.60  ...      0          2020           6\n",
              "\n",
              "[5 rows x 11 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5iSh5EhR4xKg",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 432
        },
        "outputId": "f6c765b1-58c4-411d-8f8a-cae18ca2b3e1"
      },
      "source": [
        "#getting dummies for these columns with help of pandas library\n",
        "dataset=pd.get_dummies(dataset,columns=['Fuel_Type','Transmission','Seller_Type'],drop_first=True)\n",
        "\n",
        "#dropping the columns which are redundant and irrelevant\n",
        "dataset.drop(columns=['Year'],inplace=True)\n",
        "dataset.drop(columns=['Current_Year'],inplace=True)\n",
        "dataset.drop(columns=['Car_Name'],inplace=True)\n",
        "\n",
        "#check out the dataset with new changes\n",
        "dataset.head()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "error",
          "ename": "KeyError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-18-4647f7e11c97>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#getting dummies for these columns with help of pandas library\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mdataset\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_dummies\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Fuel_Type'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'Transmission'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'Seller_Type'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mdrop_first\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;31m#dropping the columns which are redundant and irrelevant\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mdataset\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Year'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0minplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/reshape/reshape.py\u001b[0m in \u001b[0;36mget_dummies\u001b[0;34m(data, prefix, prefix_sep, dummy_na, columns, sparse, drop_first, dtype)\u001b[0m\n\u001b[1;32m    841\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Input must be a list-like for parameter `columns`\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    842\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 843\u001b[0;31m             \u001b[0mdata_to_encode\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    844\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    845\u001b[0m         \u001b[0;31m# validate prefixes and separator to avoid silently dropping cols\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   2910\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mis_iterator\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2911\u001b[0m                 \u001b[0mkey\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2912\u001b[0;31m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_listlike_indexer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mraise_missing\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2913\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2914\u001b[0m         \u001b[0;31m# take() does not accept boolean indexers\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py\u001b[0m in \u001b[0;36m_get_listlike_indexer\u001b[0;34m(self, key, axis, raise_missing)\u001b[0m\n\u001b[1;32m   1252\u001b[0m             \u001b[0mkeyarr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindexer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnew_indexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0max\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_reindex_non_unique\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkeyarr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1253\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1254\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_validate_read_indexer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkeyarr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindexer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mraise_missing\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mraise_missing\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1255\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mkeyarr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindexer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1256\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py\u001b[0m in \u001b[0;36m_validate_read_indexer\u001b[0;34m(self, key, indexer, axis, raise_missing)\u001b[0m\n\u001b[1;32m   1296\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mmissing\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1297\u001b[0m                 \u001b[0maxis_name\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_axis_name\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1298\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"None of [{key}] are in the [{axis_name}]\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1299\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1300\u001b[0m             \u001b[0;31m# We (temporarily) allow for some missing keys with .loc, except in\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyError\u001b[0m: \"None of [Index(['Fuel_Type', 'Transmission', 'Seller_Type'], dtype='object')] are in the [columns]\""
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w3LeUPI44xKh"
      },
      "source": [
        "<ul>Fuel_Type feature:\n",
        "    <li>Fuel is Petrol if Fuel_type_diesel = 0 ,Fuel_Type_Petrol = 1</li>\n",
        "    <li>Fuel is Diesel if Fuel_type_diesel = 1 ,Fuel_Type_Petrol = 0</li>\n",
        "    <li>Fuel is cng if Fuel_type_diesel = 0 ,Fuel_Type_Petrol = 0</li>\n",
        "   </ul>\n",
        "<ul>Transmission feature:\n",
        "    <li>transmission is manual if Transmission_Manual = 1</li> \n",
        "    <li>transmission is automatic if Transmission_Manual = 0</li></ul>\n",
        "<ul>Seller_Type feature:\n",
        "    <li>Seller_Type is Individual if Seller_Type_Individual = 1 </li> \n",
        "    <li>Seller_Type is dealer if Seller_Type_Individual = 0</li> </ul>\n",
        "    \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TgcXd_zg4xKi"
      },
      "source": [
        "### Pairplot"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "_GvvC2sI4xKj",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 766
        },
        "outputId": "896b6f5e-9155-4336-a207-fe9b79341cf6"
      },
      "source": [
        "#to see pairwise relationships on our dataset we will check pairplot from seaborn library\n",
        "sns.pairplot(dataset)"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.PairGrid at 0x7fa929aa40d0>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABjcAAAY4CAYAAADS6J/qAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzde3xU5Z0/8M8z92RyJQlJBEOMBNEEgjRauxW3BW2pPxS8Ye1Wty6W3d9Wg9Jt3e0W+aG2Xa2llWovWFur226h0nphldaKrbhVu2ABiahgDBgMuRFymWSu5/n9MZdkMmeSyWTmzDmTz/v1mhcmmZnzxHzm+5xznnOeR0gpQUREREREREREREREZBSmTDeAiIiIiIiIiIiIiIhoMji4QUREREREREREREREhsLBDSIiIiIiIiIiIiIiMhQObhARERERERERERERkaFwcIOIiIiIiIiIiIiIiAyFgxtERERERERERERERGQoHNwYZfny5RIAH3xo9Zg0ZpSPDDwmjTnlIwOPSWNO+dD4MWnMKB8ZeEwac8pHBh6TxpzyofEjKcwpHxo/Jo0Z5SMDj4RwcGOU7u7uTDeBaFzMKBkBc0pGwJyS3jGjZATMKRkBc0pGwJyS3jGjpFcc3CAiIiIiIiIiIiIiIkPh4AYRERERERERERERERmK4Qc3hBAOIcRfhBAHhBDNQohNoe8/JoR4XwixP/RYlOm2EhERERERERERERHR1Fky3YAU8ABYKqUcFEJYAbwihHg+9LOvSCmfzGDbiIh0S1EkWntc6Oh3o7zAgeoSJ0wmkelmTUo2/A6kH8wTUXbhZ5rIOLT+vLI+EFG20rK+sZaSHhh+cENKKQEMhr60hh4Jr6hORDQdKYrEruaTWL99P9w+BQ6rCZtXL8LyugrD7Ixkw+9A+sE8EWUXfqaJjEPrzyvrAxFlKy3rG2sp6YXhp6UCACGEWQixH0AngBeklK+HfvQNIcRBIcR3hRD2DDaRiEhXWntckZ0QAHD7FKzfvh+tPa4Mtyxx2fA7kH4wT0TZhZ9pIuPQ+vPK+kBE2UrL+sZaSnqRFYMbUsqAlHIRgNkALhRC1AP4NwDzAVwAYAaAO9VeK4RYK4TYK4TY29XVpVmbiRLFjFI6dPS7IzshYW6fgs4Bd1Lvl4mcpvp3oOw3Xk6ZJ9ID9vmpw890+jCnlGrp+LyyzycjYD2lVNPyOJ+1lPQiKwY3wqSUpwG8BGC5lLJdBnkA/AzAhXFes1VK2SilbCwrK9OyuUQJYUYpHcoLHHBYo7sAh9WEmfmOpN4vEzlN9e9A2W+8nDJPpAfs81OHn+n0YU4p1dLxeWWfT0bAekqppuVxPmsp6YXhBzeEEGVCiKLQf+cAuAzA20KIytD3BIBVAA5lrpVERPpSXeLE5tWLIjsj4fkxq0ucGW5Z4rLhdyD9YJ6Isgs/00TGofXnlfWBiLKVlvWNtZT0wvALigOoBPBzIYQZwcGa7VLKnUKI3UKIMgACwH4A/5TJRhIR6YnJJLC8rgLzm5agc8CNmfkOVJc4DbXwVzb8DqQfzBNRduFnmsg4tP68sj4QUbbSsr6xlpJeGH5wQ0p5EMD5Kt9fmoHmEBEZhskkUFOWh5qyvEw3JWnZ8DuQfjBPRNmFn2ki49D688r6QETZSsv6xlpKemD4wQ0iym6KItHa40JHvxvlBbwSwOj49yQ9Yz4pWzDLRGQ0Wtct1kkyAuaU9I4ZJT3g4AYR6ZaiSOxqPon12/fD7VMiczgur6tgh2lA/HuSnjGflC2YZSIyGq3rFuskGQFzSnrHjJJeGH5BcSLKXq09rkhHCQBun4L12/ejtceV4ZZRMvj3JD1jPilbMMtEZDRa1y3WSTIC5pT0jhklveDgBhHpVke/O9JRhrl9CjoH3BlqEU0F/56kZ8wnZQtmmYiMRuu6xTpJRsCckt4xo6QXHNwgIt0qL3DAYY0uUw6rCTPzHRlqEU0F/56kZ8wnZQtmmYiMRuu6xTpJRsCckt4xo6QXHNwgIt2qLnFi8+pFkQ4zPIdjdYkzwy2jZPDvSXrGfFK2YJaJyGi0rlusk2QEzCnpHTNKesEFxYlIt0wmgeV1FZjftASdA27MzHegusTJxakMin9P0jPmk7IFs0xERqN13WKdJCNgTknvmFHSCw5uEJGumUwCNWV5qCnLy3RTKAX49yQ9Yz4pWzDLRGQ0Wtct1kkyAuaU9I4ZJT3gtFRERERERERERERERGQoHNwgIiIiIiIiIiIiIiJD4eAGEREREREREREREREZCgc3iIiIiIiIiIiIiIjIUAw/uCGEcAgh/iKEOCCEaBZCbAp9/ywhxOtCiKNCiG1CCFum20pERERERERERERERFNn+MENAB4AS6WUDQAWAVguhLgIwH0AviulnAugF8CaDLaRiIiIiIiIiIiIiIhSxPCDGzJoMPSlNfSQAJYCeDL0/Z8DWJWB5hERERERERERERERUYoZfnADAIQQZiHEfgCdAF4A8B6A01JKf+gpbQBmZap9RERERERERERERESUOlkxuCGlDEgpFwGYDeBCAPMTfa0QYq0QYq8QYm9XV1fa2kiULGaUjIA5JSNgTknvmFEyAuaUjIA5JSNgTknvmFEygqwY3AiTUp4G8BKAjwEoEkJYQj+aDeBEnNdslVI2Sikby8rKNGopUeKYUTIC5pSMgDklvWNGyQiYUzIC5pSMgDklvWNGyQgMP7ghhCgTQhSF/jsHwGUADiM4yHFt6Gl/D+DpzLSQiIiIiIiIiIiIiIhSyTLxU3SvEsDPhRBmBAdrtkspdwoh3gLwKyHEvQD+CuDRTDaSiIiIiIiIiIiIiIhSw/CDG1LKgwDOV/l+C4LrbxARERERERERERERURYx/LRUREREREREREREREQ0vXBwg4iIiIiIiIiIiIiIDIWDG0REREREREREREREZCgc3CAiIiIiIiIiIiIiIkPh4AYRERERERERERERERkKBzeIiIiIiIiIiIiIiMhQOLhBRERERERERERERESGwsENIiIiIiIiIiIiIiIyFA5uEBERERERERERERGRoXBwg4iIiIiIiIiIiIiIDIWDG0REREREREREREREZCgc3CAiIiIiIiIiIiIiIkMx/OCGEOJMIcRLQoi3hBDNQoh1oe//PyHECSHE/tDj8ky3lYiIiIiIiIiIiIiIps6S6QakgB/Al6WUbwgh8gHsE0K8EPrZd6WUD2SwbURERERERERERERElGKGH9yQUrYDaA/994AQ4jCAWZltFRERERERERERERERpYvhp6UaTQhRDeB8AK+HvnWrEOKgEOKnQojijDWMiIiIiIiIiIiIiIhSJmsGN4QQeQB2ALhdStkP4IcAzgawCME7O74T53VrhRB7hRB7u7q6NGsvUaKYUTIC5pSMgDklvWNGyQiYUzIC5pSMgDklvWNGyQiyYnBDCGFFcGDjF1LK3wCAlLJDShmQUioAHgFwodprpZRbpZSNUsrGsrIy7RpNlCBmlIyAOSUjYE5J75hRMgLmlIyAOSUjYE5J75hRMgJdDW4IIXKFEBuEEI+Evq4VQqyY4DUCwKMA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qtlRc4VNKrWmY3dEOY3fB1IbxiyNdHD/57j6Z++N+Xlrv3FWGkpDRPkk3885+f7+iGj8ynT9xvqUiPJVqus3VSY3AFQDeEYIoUVw9MlzkiS9IoRYAWCjJEkvAXgSwLNCiDYAxwBck67CaDQCC5urMH3pPBzud6HCfmKrvBOpGb8PpDaMWSIiypR8P+fk+/sjovEr0/Ub61Miyleprt9UmdyQJOkTAKcoPH5XxO8uAFdmqkwajUBDuQ0N5bZM7ZIoZ/H7QGrDmCUiokzJ93NOvr8/Ihq/Ml2/sT4lonyVyvpN9WtuEBERERERERERERHR+MLkBhERERERERERERERqQqTG0REREREREREREREpCpMbhARERERERERERERkaowuUFERERERERERERERKrC5AYREREREREREREREakKkxtERERERERERERERKQqTG4QEREREREREREREZGqMLlBRERERERERERERESqwuQGERERERERERERERGpCpMbRERERERERERERESkKkxuEBERERERERERERGRqjC5QUREREREREREREREqsLkBhERERERERERERERqQqTG0REREREREREREREpCpMbhARERERERERERERkaqoLrkhhJgohHhLCPGpEGKbEGKZwjZfFEL0CiE+Hvq5KxtlJSIiIiIiIiIiIiKi1NNluwBJ8AH4niRJHwkh7AA2CSHelCTp0xHbvSNJ0qIslI+IiIiIiIiIiIiIiNJIdSM3JEnqlCTpo6Hf+wFsB1CT3VIREREREREREREREVGmqC65EUkIMQnAKQA+VPjzmUKIzUKIPwghmjNaMCIiIiIiIiIiIiIiShvVJjeEEDYA6wD8iyRJfSP+/BGAekmSZgH4CYAX4rzOEiHERiHExiNHjqSvwERJYoySGjBOSQ0Yp5TrGKOkBoxTUgPGKakB45RyHWOU1ECVyQ0hhB7BxMavJEn635F/lySpT5KkgaHfXwWgF0KUKb2WJEmPS5LUKklSa3l5eVrLTZQMxiipAeOU1IBxSrmOMUpqwDglNWCckhowTinXMUZJDVSX3BBCCABPAtguSdLKGNtUDW0HIcQcBN9nd+ZKSURERERERERERERE6aLLdgGS8AUA1wLYIoT4eOixHwCoAwBJkh4DcAWAbwkhfAAGAVwjSZKUjcISEREREREREREREVFqqS65IUnSuwDEKNusBrA6MyUiIiIiIiIiIiIiIqJMUt20VERERERERERERERENL4xuUFERERERERERERERKrC5AYREREREREREREREakKkxtERERERERERERERKQqTG4QEREREREREREREZGqMLlBRERERERERERERESqwuQGERERERERERERERGpCpMbRERERERERERERESkKkxuEBERERERERERERGRqjC5QURo/W+RAAAgAElEQVREREREREREREREqsLkBhERERERERERERERqQqTG0REREREREREREREpCpMbhARERERERERERERkaowuUFERERERERERERERKrC5AYREREREREREREREakKkxtERERERERERERERKQqqkxuCCEmCiHeEkJ8KoTYJoRYprCNEEKsEkK0CSE+EUKcmo2y/n/27j++jfu+8/z7C4IkKJCAZIoiINmybEduHJKy6nKTNm2T1E5ySlaW1bRVkt1bX7ftuu21jRvt7bXd1UmxTu3dtlu1+fVom6TZxN1NGrVpFNlN3aTxdt1t2m0UR5ZI27EcR7YlEaRMSQAFEiRBfPcPEhBAAiAADgYY6PV8PPCwMPOdmS8Gb3wwg685AwAAAAAAAAAAnOVvdAdqlJb0b621TxtjeiR9yxjzNWvts3lt3iVp+9LjTZL+YOm/AAAAAAAAAADAwzz5lxvW2jFr7dNL/56S9JykLcuaPSDpUbvoHyWtN8ZEXe4qAAAAAAAAAABwmCcHN/IZY7ZJ+n5J/3PZrC2SXs17fl4rB0AAAAAAAAAAAIDHeHpwwxjTLemLkn7VWpuocR0PGWNOGmNOXrp0ydkOAg4go/ACcgovIKdodmQUXkBO4QXkFF5ATtHsyCi8wLODG8aYdi0ObPxXa+1fFGlyQdItec9vXppWwFr7CWvtsLV2uK+vrz6dBdaAjMILyCm8gJyi2ZFReAE5hReQU3gBOUWzI6PwAk8ObhhjjKQ/lvSctfZoiWYnJD1oFv2gpLi1dsy1TgIAAAAAAAAAgLrwN7oDNfphSf9K0hljzKmlaf9e0lZJstb+oaSvSHq3pBclTUv61w3oJwAAAAAAAAAAcJgnBzestf9DklmljZX0S+70CAAAAAAAAAAAuMWTl6UCAAAAAAAAAAA3LgY3AAAAAAAAAACApzC4AQAAAAAAAAAAPIXBDQAAAAAAAAAA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\n",
            "text/plain": [
              "<Figure size 1620x1620 with 90 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2fQG7QT54xKk"
      },
      "source": [
        "### Heat map"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c0x5y_O84xKk",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 708
        },
        "outputId": "d450071f-c6c9-4997-a91e-094b2bca463e"
      },
      "source": [
        "#create correlation matrix\n",
        "correlations = dataset.corr()\n",
        "indx=correlations.index\n",
        "\n",
        "#plot this correlation for clear visualisation\n",
        "plt.figure(figsize=(26,22))\n",
        "#annot = True , dsiplays text over the cells.\n",
        "#cmap = \"YlGnBu\" is nothing but adjustment of colors for our heatmap\n",
        "sns.heatmap(dataset[indx].corr(),annot=True,cmap=\"YlGnBu\")\n",
        "#amount of darkness shows how our features are correalated with each other \n"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fa91f289350>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABVIAAATmCAYAAADDb1qpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3xddf3H8dc3aVb3HnQX2lJaRkvLHjIFZIgKiIjgYIjiQH4gILKqIAqigEJZIiJ7DwGBshFoS6GDbrpnmqbpyM7390dC29BxE8hNTtvX8/HIIznnfO/N5/SeniTv+x0hxogkSZIkSZIkafMymroASZIkSZIkSUo6g1RJkiRJkiRJSsEgVZIkSZIkSZJSMEiVJEmSJEmSpBQMUiVJkiRJkiQpBYNUSZIkSZIkSUrBIFWSJEmSJElS4oQQ7g4hLA0hTNzM8RBC+GsIYUYI4eMQwrANjp0RQphe83FGQ9RjkCpJkiRJkiQpif4BHLWF40cD/Ws+zgb+DhBCaA9cAewN7AVcEUJo92WLMUiVJEmSJEmSlDgxxjeAgi00OQH4Z6z2P6BtCKEb8FXgvzHGghjjCuC/bDmQrRODVEmSJEmSJElbo+7AvA2259fs29z+L6XZl32CusjrdWpsjO+j5GrdokdTl6AmtvMfv97UJagJdW9Z0dQlqIn1b13e1CWoid1/3ZKmLkFN7LHb2jR1CWpik1Y0yp+fSrAh7fydULBb+2NDU9eQZNtbhlYy78FzqB6S/5lRMcZRTVVPKv4kkyRJkiRJktToakLTLxOcLgB6brDdo2bfAuArn9v/2pf4PoBD+yVJkiRJkiRtnZ4Gvheq7QOsjDEuAl4EjgwhtKtZZOrImn1fij1SJUmSJEmSJCVOCOEBqnuWdgwhzAeuALIAYoy3Ac8DxwAzgLXA92uOFYQQrgE+qHmqq2OMW1q0qk4MUiVJkiRJkiQlTozx1BTHI/CTzRy7G7i7IesxSJUkSZIkSZISIARn4UwyXx1JkiRJkiRJSsEgVZIkSZIkSZJSMEiVJEmSJEmSpBQMUiVJkiRJkiQpBRebkiRJkiRJkhIg2Ocx0Xx1JEmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJSMEiVJEmSJEmSpBRcbEqSJEmSJElKgBDs85hkvjqSJEmSJEmSlIJBqiRJkiRJkiSlYJAqSZIkSZIkSSk4R6okSZIkSZKUAM6Rmmy+OpIkSZIkSZKUgkGqJEmSJEmSJKVgkCpJkiRJkiRJKRikSpIkSZIkSVIKLjYlSZIkSZIkJUAIoalL0BbYI1WSJEmSJEmSUjBIlSRJkiRJkqQUDFIlSZIkSZIkKQWDVEmSJEmSJElKwcWmJEmSJEmSpESwz2OS+epIkiRJkiRJUgoGqZIkSZIkSZKUgkGqJEmSJEmSJKXgHKmSJEmSJElSAoRgn8ck89WRJEmSJEmSpBQMUiVJkiRJkiQpBYNUSZIkSZIkSUrBIFWSJEmSJEmSUnCxKUmSJEmSJCkBXGwq2Xx1JEmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJSMEiVJEmSJEmSpBRcbEqSJEmSJElKgGCfx0Tz1ZEkSZIkSZKkFAxSJUmSJEmSJCkFg1RJkiRJkiRJSsE5UiVJkiRJkqQECME+j0nmqyNJkiRJkiRJKRikSpIkSZIkSVIKBqmSJEmSJEmSlIJBqiRJkiRJkiSl4GJTkiRJkiRJUgK42FSy+epIkiRJkiRJUgoGqZIkSZIkSZKUgkGqJEmSJEmSJKVgkCpJkiRJkiRJKbjYlCRJkiRJkpQALjaVbL46kiRJkiRJkpSCQaokSZIkSZIkpWCQKkmSJEmSJEkpOEeqJEmSJEmSlACB0NQlaAsMUtPstj+ew9GHDWXZ8iKGH3FRU5ejNDjkgIGMvPR4MjMyuP/R97n5ztG1jnfv1pabrz2F1q3yyMzMYOSNz/PKG1No1iyDG685id126U5mZgaPPDWWv94xejPfRUmzV6e2nD+4HxkBnpu7hH/PXFDreFZG4NI9BjCgTQuKyiq4atxUFheXMrxjG87euQ9ZGYHyqsjfP5nNh8tXAnBIt46c3r8HGSHw7pICbp8ypylOTXUUY2TeQw9RNHECGdnZ9DnzTJr36r1RuzVz5jD7H/cQy8tpPWRXep5yCiEE1s6bx9z7/0VVeTkhI5Ne3/kOLfr2pbJ4LZ/edTdlKwqIlZV0OeJIOu6/fxOcoeojxsi4ex9h0fhJZGZns/ePT6d931612lSUlvH2TXeyemk+IQS677kru5/69Vpt5r33IW/fdCdHjryI9jtufD0puQ7avRu/+f5wMjMCD78yg9ufmlzr+DcO7sevTx/K4oK1APzrhWk8/OpMAKY+eCpT5xYCsCh/Ledc/3rjFq8GEWPknj8/yYfvfEJObjbnXf5t+g3ssVG7K8/7GyuWF5GdkwXAb246mzbtW607/r/RH3Pjpfdy7d2/YMdBPRutfn15MUZeuP1xpn8wmaycLL5+wWl022nj1/Bfl/+d1QVFVFVW0WtwP4457yQyMjN49Np/kL9gKQAlq4vJbZnHubf4N+TW5LP7wLia+8BPNnMfuOJz94HLa+4Do597n/tueZb2ndoAcPS39uew4/dp1HOQtGUGqWl23yOvc9u9L3Lnn89r6lKUBhkZgesuP5GTfziKhUtW8uLDP+PF0ZOYNnPpuja/PPcwnnrhY+598F0G7NiZ+2//ISMOv5bjv7obOdnN+MoJN5KXm8Ubz17IE8+NZ97CFU14RqqLDOAXQ/rxq/cmsay4jNsP3J23lxQwZ3XxujZf69mFVeUVnDZ6HIfu0JFzBvXhqnFTWVlWwSUffMLy0jL6tmrOH/fehW+9PIbWWc348S59OOvN8dVtdu/PsA5tGFcTsip5iiZOpHTpEgZfM5I1n37KnPvvZ9All27Ubu6/76f36d+jRd++zLj5rxRNmkibIbsy/7FH6XbssbQZsisrJ0xg/uOPMfBXF7J09GvkduvGTj/9KeWrVjHpt5fTfu+9yWjmj+wkWzR+EqsXL+Nrf76S5TNmM+auBzly5MZ//O587OF0GTyAyooKRo/8KwvHT2KHPQYDUF5cwrQXRtNhpz6NXL2+rIwQuPKHIzhj5KssXr6Wx689ilfGzGfGgqJa7Z57Zw5X3T1mo8eXlFVy/EX/aaxylSYfvjuFxfPy+esjlzB90lzuvP4xfn/XzzfZ9mdXnrbJkLR4TQn/efhN+g/utYlHKelmjJlMwYJlnH/nb1gwdQ7P3fIIP7rpgo3anXTJ98lpnkuMkUd+dzeT3xrPkIOH8a1LzlzX5sU7niC3RV4jVq+G8OG7U1g0L5+ba+4Dd1z/GNdu5j7w883cB/Y7bA9+dOE30l2qpC/IOVLT7O33p1BQuLqpy1CaDNutF5/OzWfO/ALKyyt58vnxHHXo4FptYoRWLXMAaN0qjyVLi9btb56XTWZmBrm5WZSXV7JqTUmjn4Pqb1DbVixYU8KitaVUxMirC5ZxQJf2tdrs36U9L86rDtRfX5TPsI7V7ypPL1rD8tIyAD5dtZacjAyyMgI7NM9l/ppiVpZVADA2v5CDu3VoxLNSfRV+NJ4O++xLCIGW/fpRWVxM+crCWm3KVxZSWVxMy379CCHQYZ99KRw/HoAQApXF1f/nK4uLyWrTlpoDVJWWEGOkqrSUZi1aEDL8cZ10C8Z+TJ8D9yaEQMf+fSlfW0zxitpvhDTLyabL4AEAZDZrRvu+PSlevv6amfDwsww67kgysrIatXZ9ebvv1IE5i1cxb+lqyiureO6dORw+wp6E25sxb0zkoKP3JITAgCG9WbO6mBX5RakfuIGHRr3ACd89hKxs7wNboyn/m8huh40ghECPnftQsqaYVQUbvyme0zwXgKrKKiorKjc6HmNk8pvV4aq2Lh+8MZGDv+R9QFKy1bl7SwihOfAroFeM8awQQn9gYIzx2bRVJyVc186tWbh4/R/BC5esZNhutXsQ/PHWl3j4zrP44Wn70zwvm5N+MAqAZ176mKMOG8zHb1xO89xsfnvd0xSuLEbJ1zEvm6UlZeu2l5WUMahdq9ptcrNZWlIKQGWENeUVtMlqxsryinVtDu7WgWkr11BeFZm/tpieLfLompfDspJSDujanizDs0QrLywku327ddvZbdtRtqJwfSAKlK0oJLvd+jZZ7dpRXlh9z+hx8ilM/8tNzH/sUYiRgRddDEDnQw5hxq238PFF/0dVaSn9zjrLIHUrUFywkuYd1r/2ee3bUlxQSF67NptsX7ZmLQvGTWDAUYcAUPDpXNYWrGCHYUP45NmXG6VmNZwu7fNYtHztuu3Fy9eye/+N3wz76t69GDGoM7MXreJ3945d95icrEyeuPYoKisjtz01iZc/mN9otavhFCxbSccu6+8DHTq1oWDZStp1bL1R27+NfJCMzAz2/spufPP7hxNCYNbU+eQvLWTY/rvw9P2vNWLlaiir8gtp02n9NdC6YxtW5a+kVfuNfxb86zd/Z8G0Oey05yB2OWCPWsfmTpxJi7at6NC9c9prVsMqWLaSDnW8D9xacx/YZ4P7AMB7r33MJ+Nn0a1XJ878+fF07NJuo8dKajr1GSd4DzAW2LdmewHwCGCQKm3BiccM5cEnxnDbP95g+B69ueUPp3Lw8TcwdNdeVFZWsfvB19C2dR5P/es83nh3OnPmFzR1yWoEfVrmcc7Ovbnwveo59FaXV/LnCTO5YthAIpGJBavYoUVuE1epdFr2+uv0PPlk2g3bk4IxY5jzz3sZ8MsLKJo0ieY9ezLggl9RumwZ02/6M7vs1J/MPIf3bSuqKit59+Z7GPDVr9CyS0diVRUf3vc4e//49KYuTWn06tj5PPv2bMoqqvj24Ttx/U/25fSrXwHg4POeZMmKYnp2bsl9vz2MaXMLmbvEEU3bqp9deRrtO7eheE0JN1x6L2/8ZywHHjWMf/7lac67/NtNXZ4ayXdH/piKsnIev/6ffPrRNHYctvO6YxNeH8eQr9gbdVv2sytPo0PNfeBPNfeBg48ZzvADBnPAEcPIym7Gf594l1uueZArb/lxU5erRhaCnSiSrD5B6o4xxlNCCKcCxBjXhs/eMtmEEMLZwNkAzdoNp1nLnb5cpVICLV5axA5d17/juEOXNixeUnv4zne+NYJTz7oTgDHj55Cb04wO7ZrzjWOH8upbU6moqCK/YA0fjJvN7kN6GKRuBfKLy+icm71uu1NuNvnFpbXblJTROTeHZSVlZAZosUFv1E652YwcPojfj5/OwrXrp3N4Z+kK3llaPUfucb26UEVshLNRfSwdPZr8t94EoEWfPpQVrJ/TuKxwBdnt2tZqn92uLWUr1rcpX7GCrLbVbZa/+w49TzkFgHZ77smc+/4JQP47b9P1qKMJIZDbuTM5HTtSsngxLfr2Teu5qf6mv/Q6M199G4D2/XqzdoNh+sUFheS1b7vJx31wx79p2bUTA485FIDyklJWzlvIq1ffBEDJyiLe+NPtHHThOS44tZVYUlBMtw7N12137dCcJQW1R5kUrl4/kuHhV2Zy8XeHrn/8iuq285au5r3JS9ilTzuD1K3EC4++xStPvwfAjoN6kr9k/X1g+bKV6xaM2VD7ztX78lrkcsCRQ5kxeS4jDhrMvFmLuOq8vwFQWLCK6y+6m4uu/4ELTiXc+8+8ybgX3wVgh/69WLls/TVQlL+SVh03PTIBoFl2FgP33ZWp/5u4LkitqqxkyjsfcfZf/y+9havBvPDoW7xccx/YaVBPltfhPtDhc/eB6ZPncvAxw2nVpsW6Nocevzf33Wq/NSlp6hOkloUQ8qD6L/sQwo5A6eYaxxhHAaMA8nqdahqgbdKHE+bRr3dHenVvx6KlRXz9mD348f/9u1abBQsLOXCf/jz05Bj69+tMTk4z8gvWsGDRCg7YeycefXoczfOyGLZ7b0b9860mOhPVx5SVq+hRMww/v6SMQ7t34ppxU2u1eXtJAV/t2ZlJhas4uFtHPsyvDthbNsvkur124fYps5m4YlWtx7TNzqKwrJyWWZmc0LsrV37uOdX0Oh9yCJ0PqR6KvXLCxywdPZp2I0aw5tNPyczLqzWsHyCrTVsy8/JYPWsWLfr2Zfn/3qXzIdXhWXbbtqyeNo1WAweyasoUcjtXD9/Lbt+BVVM+oVX//pQXFVGyZAk5nTo27omqTvofeTD9jzwYgIXjJjL9pdfptd+eLJ8xm6zmeZsc1v/xQ89QXlzCXmeftm5fdvM8vnHH9eu2X7n6JoaedqIh6lbk45nL6d2tFT06tWBJQTFf2683F/z17VptOrXNZVlh9Ztnhw3vzsz51XPmtW6RTUlpBWUVVbRrlcOeAztxx1OTG/0c9MUc9a0DOOpbBwAw7u3JvPDo2+x/xFCmT5pL8xa5Gw3nrayoZM3qYlq3bUlFRSVj3/6EXYf3p3nLPO564Zp17a4872+cfv5xhqhbgb2OO5C9jjsQgGnvT+KDZ95kyMHDWDB1Djktcjca1l9WXEppcQmt2rehqrKS6e9PpteQfuuOz/pwGh17dKF1x02/Gafk2fA+MPYL3gd2G94fgBX5Revaj3lzEj36OL2DlDT1CVKvAF4AeoYQ7gf2B85MR1HbkntvPp8D9x1Ex3atmPHeLVxz46Pc+9BrTV2WGkhlZRWXjHySB+88i8yMDB54/H2mzljCRecfyUcT5/Pi6Mlcef0z3HD1SZxzxoHECD+75GEA7v73O/zldyfz+jO/IhB48IkPmDxtUROfkeqiMsJNk2bxp70HkxHg+XlLmb26mB8M6MWUlat5Z0kBz89bwmV7DOD+Q4axqryCq2pC0RP7dKN781zO6N+TM/pX/3F04XuTKSwr52eD+7Jj6+p3oe+dPo/5Lj6WaK2H7MrKCROZ+JvLyMjOps8ZZ647Nvmaq9nl8t8C0OvU7zD73n9QVVZGmyFDaD1kCAC9Tz+deQ89RKyqIjRrRq/vVg/r7va1rzH7H/cw6aorAeh+4jdo1rL2HLxKnm5DB7Nw/CSe/cWVNMvJZu9zvrvu2Au//j1HXXcpa5evYPKTL9B6hy68eOl1QHUYu+Oh+zdV2WoglVWRq+4ewz2XHUpmRuCR0TOZPn8lPz95NybOXM4rYxdwxtE7c9jw7lRURlauLuWiv1X3YNuxe2tGnr03VVWRjIzA7U9OZsYCFybZGg3dbxDj3vmEn510Ldk5WZz3m/XD9P/vezfwx3/+ivLyCn73izuorKikqqqKXUcM4PAT9mnCqtWQ+o/YhekfTObmH15DVk42J/zyO+uO3fbT6zn3losoKynlwavuoKK8ghgjfXbrz/Bj1v8cmPjGOBeZ2ooN228QH77zCefX3Ad+ssF94MLv3cCfau4DIz93Hzis5j7w/MNvMuatSWRmZtCydfNaj5eUDCHGuncWDSF0APYBAvC/GGN+XR5nj1S1btGjqUtQE9v5j19v6hLUhLq3rEjdSNu0/q3Lm7oENbH7r1vS1CWoiT122+aHOGv7MGlFffrxaFs0pJ2/Ewp2a3/sZqeJFHTe+VfbVYa2dMoNW9X1UOefZCGEE4FXY4zP1Wy3DSF8Pcb4ZNqqkyRJkiRJkrYTLjaVbPV5da6IMa5bRSfGWEj1cH9JkiRJkiRJ2qbVJ0jdVFvHZkiSJEmSJEna5tUnSB0TQrgxhLBjzceNwNh0FSZJkiRJkiRJSVGfHqXnA5cDD9Vs/xf4SYNXJEmSJEmSJG2HnCM12eocpMYY1wC/TmMtkiRJkiRJkpRIKYPUEMJNMcZfhBCeAeLnj8cYj09LZZIkSZIkSZKUEHXpkXpfzec/pbMQSZIkSZIkSUqqlEFqjHFsCCETODvGeFoj1CRJkiRJkiRJiVKnOVJjjJUhhN4hhOwYY1m6i5IkSZIkSZK2Py42lWR1XmwKmAW8HUJ4Gljz2c4Y440NXpUkSZIkSZIkJUh9gtSZNR8ZQKv0lCNJkiRJkiRJyVOnIDWEsAcwCZgUY/wkvSVJkiRJkiRJUrKknHghhPBb4GHgm8BzIYSz0l6VJEmSJEmSJCVIXXqkngLsEWNcG0LoALwA3JHesiRJkiRJkqTtSwguNpVkdXl1SmOMawFijMvr+BhJkiRJkiRJ2mbUpUdqvxDC0zVfB2DHDbaJMR6flsokSZIkSZIkKSHqEqSe8LntP6WjEEmSJEmSJElKqpRBaozx9bo8UQjhsRjjN798SZIkSZIkSdL2xzlSk60hX51+DfhckiRJkiRJkpQYDRmkxgZ8LkmSJEmSJElKDPsLS5IkSZIkSVIKDRmkhgZ8LkmSJEmSJElKjJSLTdXDxQ34XJIkSZIkSdJ2JTh4PNHqHKSGECaw8TyoK4ExwMgY40sNWZgkSZIkSZIkJUV9eqT+B6gE/l2z/W2gObAY+AdwXINWJkmSJEmSJEkJUZ8g9fAY47ANtieEEMbFGIeFEL7b0IVJkiRJkiRJUlLUZ+KFzBDCXp9thBBGAJk1mxUNWpUkSZIkSZIkJUh9eqT+CLg7hNASCEAR8KMQQgvg2nQUJ0mSJEmSJG0vQnCxqSSrc5AaY/wA2DWE0KZme+UGhx9u6MIkSZIkSZIkKSnqHKSGEHKAbwJ9gGYhBABijFenpTJJkiRJkiRJSoj6DO1/ClgJjAVK01OOJEmSJEmSJCVPfYLUHjHGo9JWiSRJkiRJkrQd+2wEuJKpPjPYvhNC2DVtlUiSJEmSJElSQtWnR+oBwJkhhE+pHtofgBhj3C0tlUmSJEmSJElSQtQnSD06bVVIkiRJkiRJUoKlDFJDCK1jjEXAqkaoR5IkSZIkSZISpy49Uv8NHAuMBSLVQ/o/E4F+aahLkiRJkiRJ2q6EUJ/ljNTYUgapMcZjaz73TX85kiRJkiRJkpQ8dRnaP2xLx2OM4xquHEmSJEmSJElKnroM7b9hC8cicGgD1SJJkiRJkiRJiVSXof2HNEYhkiRJkiRJkpRUdRna/40tHY8xPt5w5UiSJEmSJEnbp4CLTSVZXYb2H7eFYxEwSJUkSZIkSZK0TavL0P7vN0YhkiRJkiRJkpRUde4vHELoEkK4K4Twn5rtXUIIP0xfaZIkSZIkSZKUDHUZ2v+ZfwD3AJfVbE8DHgLuauCaJEmSJEmSpO1OCM6RmmT1eXU6xhgfBqoAYowVQGVaqpIkSZIkSZKkBKlPkLomhNCB6gWmCCHsA6xMS1WSJEmSJEmSlCD1Gdp/AfA0sGMI4W2gE/CttFQlSZIkSZIkSQmSskdqCGFECKFrjHEccDBwKVAKvATMT3N9kiRJkiRJktTk6tIj9Xbg8Jqv96N6sanzgT2AUdgrVZIkSZIkSfrSXGwq2eoSpGbGGAtqvj4FGBVjfAx4LIQwPn2lSZIkSZIkSVIy1CXmzgwhfBa4Hga8usGx+syxKkmSJEmSJElbpboEoQ8Ar4cQ8oFi4E2AEMJOwMo01iZJkiRJkiRJiZAySI0x/i6E8ArQDXgpxhhrDmVQPVeqJEmSJEmSJG3T6jQ0P8b4v03sm9bw5UiSJEmSJEnbp1CnWTjVVHx1JEmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJSqNMcqZIkSZIkSZLSLNjnMcl8dSRJkiRJkiQphUbpkdq6RY/G+DZKsKI185u6BDWxhQsqm7oENaFmvRwAsb2bV5TZ1CWoiVW1zWnqEtTEispCU5egJlZe5TWwvSvzGpC0lbNHqiRJkiRJkiSlYJAqSZIkSZIkSSk41lKSJEmSJElKgOBiU4nmqyNJkiRJkiRJKRikSpIkSZIkSVIKBqmSJEmSJEmSlIJBqiRJkiRJkiSl4GJTkiRJkiRJUgKEEJq6BG2BPVIlSZIkSZIkKQWDVEmSJEmSJElKwSBVkiRJkiRJklIwSJUkSZIkSZKkFFxsSpIkSZIkSUqAYJ/HRPPVkSRJkiRJkpRIIYSjQghTQwgzQgi/3sTxP4cQxtd8TAshFG5wrHKDY09/2VrskSpJkiRJkiQpcUIImcCtwBHAfOCDEMLTMcbJn7WJMf5yg/bnA0M3eIriGOMeDVWPPVIlSZIkSZIkJdFewIwY46wYYxnwIHDCFtqfCjyQrmLskSpJkiRJkiQlQAj2efyc7sC8DbbnA3tvqmEIoTfQF3h1g925IYQxQAVwXYzxyS9TjEGqJEmSJEmSpEYXQjgbOHuDXaNijKO+4NN9G3g0xli5wb7eMcYFIYR+wKshhAkxxplftF6DVEmSJEmSJEmNriY03VJwugDoucF2j5p9m/Jt4Cefe/4FNZ9nhRBeo3r+1C8cpNpfWJIkSZIkSVISfQD0DyH0DSFkUx2WPv35RiGEnYF2wLsb7GsXQsip+bojsD8w+fOPrQ97pEqSJEmSJElKnBhjRQjhp8CLQCZwd4xxUgjhamBMjPGzUPXbwIMxxrjBwwcBt4cQqqjuTHpdjNEgVZIkSZIkSdrqhdDUFSROjPF54PnP7fvt57av3MTj3gF2bchaHNovSZIkSZIkSSkYpEqSJEmSJElSCgapkiRJkiRJkpSCQaokSZIkSZIkpeBiU5IkSZIkSVIS2OUx0Xx5JEmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJScI5USZIkSZIkKQlCaOoKtAX2SJUkSZIkSZKkFAxSJUmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJScLEpSZIkSZIkKQlcbCrR7JEqSZIkSZIkSSkYpEqSJEmSJElSCgapkiRJkiRJkpSCQaokSZIkSZIkpeBiU5IkSZIkSVIS2OUx0Xx5JEmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJScI5USZIkSZIkKQFiCE1dgrbAHqmSJEmSJEmSlIJBqiRJkiRJkiSlYJAqSZIkSZIkSSkYpEqSJEmSJElSCi42JUmSJEmSJCWBa00lmj1SJUmSJEmSJCkFg1RJkiRJkiRJSsEgVZIkSZIkSZJSMEiVJEmSJEmSpBRcbEqSJEmSJElKggxXm0oye6RKkiRJkiRJUgoGqZIkSZIkSZKUgkGqJEmSJEmSJKXgHKmSJEmSJElSEgTnSE0ye6RKkiRJkiRJUgoGqZIkSZIkSZKUgkP7v0KV8uYAACAASURBVKRDDhjIyEuPJzMjg/sffZ+b7xxd63j3bm25+dpTaN0qj8zMDEbe+DyvvDGFZs0yuPGak9htl+5kZmbwyFNj+esdozfzXbS1uu2P53D0YUNZtryI4Udc1NTlqBEc2LMdv9lvRzJD4OEpixk1fl6t49/ftTsnD+pKRVWkoKScS16bxsLVpU1UreprRMe2/GRQPzICPD9/CQ/OWlDreFZG4OLdBjCgdQuKyiu4ZvxUlhRXv76n9uvO0T26UBXhlk9mMSa/EIBv9tmBY3p0IRL5dNVarp8wnfKqyAm9uvLNPjvQvUUeJ778HkXlFY1+vtqyvTq15ee79iMjBJ6ds4T7Z8yvdTwrI3DZ0AEMbNuSorIKrhgzhcXF6/+/d87L4b5DhnHP1Lk8OLP6Wjq53w4c26sLEZhVtJZrx0+jrCo25mnpCzpocBd+e8oeZGQEHn7rU257Yeom2x01rDt/O3dfTvjdK0yYs4IT9urJWV8duO74zt3bcNzIl/lk/srGKl0NJMbIA399ggnvfUJ2TjY/uORUeg/osdn2N19yF8sWLefqf1T/jnjblf9kybylAKxdXUzzlnlccdeFjVK7GkaMkf+OeoyZYybTLCeb435xGl136rlRuwd/+zdWFxRRVVVFz1125Ks/PomMzAzeuP95xr/4Ls3btATgK987lp1GDG7s09CXEGPknzc9wUfvfkJ2bjbnXHYqfQdufB8Y+dNbKcwvIisnC4Bf33QObdq14pPxM/nXX55k7sxF/PSq09n7kN0b+xQkpWCQ+iVkZASuu/xETv7hKBYuWcmLD/+MF0dPYtrMpeva/PLcw3jqhY+598F3GbBjZ+6//YeMOPxajv/qbuRkN+MrJ9xIXm4Wbzx7IU88N555C1c04Rmpod33yOvcdu+L3Pnn85q6FDWCjABX7r8TZz43gcVrSnnsG0N5dfZyZhSuXddm8vLVnPj4h5RUVPGdXbpx0T59+cXLU5qwatVVBvCzwf246P1JLCsp42/77c67SwuYs7p4XZuje3RhdXkF33tjHId068hZA/swcvxUerfM45BunfjhWx/SISebP+41mDNeH0f7nGxO7N2NH7z5IWVVVVy+x0AO7daJFxcsZVLhKv73wSRu3GtI0520NisDuGC3HfnluxNZVlzGHQftwduLlzN7g+vha726sKq8glNfGcthO3Tk3F36cOXY9eHa+YP78t7S9T/3O+Zm882+O3D66HGUVVVx1Z4DOax7J/4zbylKtowAV31nKN/785ssXrGWJy89jJc/WsiMRatqtWuR04wzD92JD2ctX7fvqffn8dT71W+6DezemtvO288QdSs14b1PWDo/n9/ffymzJs/hXzc+ymW3/WKTbce+8TE5edm19p175ffWff3QrU/RvEVuWutVw5s5ZjIFC5dx7qjLWTh1Ni/87WHOvPFXG7U78dffJ6d5HjFGHr/2bj5560MGH7wnAHt9/Svs843DGrt0NZCP3v2ExfPzueGhS5kxaQ73/OlRrr5j0/eB8674Lv0G1Q7aO3ZpxzmXncpzD7zWCNVK+iIc2v8lDNutF5/OzWfO/ALKyyt58vnxHHVo7XcMY4RWLXMAaN0qjyVLi9btb56XTWZmBrm5WZSXV7JqTUmjn4PS6+33p1BQuLqpy1Aj2a1zK+YUFTNvVQnlVZHnZizjsD4darV5b+FKSiqqABi/pIiuLXKaolR9ATu3bcWCNSUsKi6lIkZGL1rGfp3b12qzX+f2vLSgOvR6fXE+wzq0Wbd/9KJllFdFFheXsmBNCTu3bQVAZgjkZGaQESA3M4P80jIAZhStWdebVckzqF3N9bC2+np4ZcEyDuha+//7gV078EJNCPraonz27Nh2g2PtWbS2hE9Xra31mMyM6ushM0BuZib5JWXpPxl9abv3bc+cpauZl7+G8srIsx/M44jdd9io3QUnDOb2F6dSWl61yec5bkQvnv1g3iaPKfnGvzWRfb86nBACOw7uw9rVxRQuL9qoXcnaUv778Osc+70jNvk8MUbGjP6IvQ4flu6S1cCmvTeBXQ/dixAC3XfuS8maYlYXbPzGSE7zPACqKquoLK8guLDMNmPsWxM58Kjq+0D/IX1Yu6qYFfkb3wc2p1O39vTaaQevie1d2M4+tjL1ClJDCAeEEL5f83WnEELf9JS1dejauTULFxeu2164ZCVdu7Sp1eaPt77Et44bxoejL+P+237ApSOfBOCZlz5mbXEZH79xOeNeuYy/3/06hSuLkbT16to8h0UbDNNfvKaULi2yN9v+Wzt35Y259kLfWnTMzWbZBqHWspIyOubmbNRmaUn1NVAVYU1FBa2zmtExN6fWY/NLyuiYm01+aRmPfLqAB74ynEcO3YvVFZWMzS9EydcpN5ulGwTdy0pK6fi53mUdN2hTWXM9tMluRl5mBt/ZqQf3TJ1bq31+SRkPzljAo0eM4Mkj92Z1RQUfLPN62Bp0bZvHooL1v8ctKiymS7u8Wm0G92pLt/Z5jJ6weLPP87URPXjmfYPUrVVhfhHtO69/w6Rdp7YULts4RHvy7v9w5MkHk52z6d8Rpn88i9btW9KlR6e01ar0WL18Ja03eNOsVYe2rFq+6R7mD1z+N/5y2qVkN89l5/33WLd/7LNvcsdPr+PZm+6nePXaTT5WyVWwrIgOG9wH2nduy4pN3AcAbv/9A1xyxp944p6XiNFpfKStRZ2D1BDCFcDFwCU1u7KAf6WjqG3JiccM5cEnxjD0kN9x2rl3c8sfTiWEwNBde1FZWcXuB1/DiCN+z7nfP4jePdqnfkJJ24Tj+3dm106tuPMj/2DenrVslsl+Xdpz2utjOPnVD8jLzODwHfzDeVv3/YG9eHjWQoora/dKbJmVyQFd23PKyx/w9ZfeJy8zkyMNUrYJIcBlJ+3O7x75eLNtdu/bnpKySqYtrHvPJW195k5fwLIFyxl20G6bbfPeyx+y12H2Rt3WnXrNefzsvpFUllcw5+NpAAw75gB+fMdv+dFfL6Jl+za8cucTTVyl0uW8K07jD/ddxG//9lOmfDSLt14Y09QlSaqj+syReiIwFBgHEGNcGEJotbnGIYSzgbMBWnU9gry2294kyYuXFrFD1/XvNu3QpQ2Ll9R+t+k73xrBqWfdCcCY8XPIzWlGh3bN+caxQ3n1ralUVFSRX7CGD8bNZvchPZgzv6BRz0FSw1m8tpRuLdf3UOzaIoclazYelrtf97acN7QX33n6IxeR2Yrkl5TRKXd976FOudnkl5Ru1KZzbg75JWVkBGjRrBlF5RXkl5TWemzH3GzyS8oY1rEti9eWsrKseiGpNxcvZ5e2rXh54bLGOSl9YctKyuict/7/e6fcHPKLa/9/z69ps6ykjMya62FlWQW7tGvFV3boyI936UPLrGbEGCmrrKKgtIxFa0sorLkeXl+0nCHtWvPSfK+HpFtcWEy39ut7oHZrm8eSFet7qLbMbcaA7q154FcHA9CpTS6jfrIfZ9/6DhPmVI9MOG5ET3ujboVefeIt3nz2fwD0GdiTgqXre5GvWFZI2061R6vNnDSb2VPncfEp11BVWUXRitVc//NbuegvPwGgsqKScW9+zOWjLmi8k9CXMubZNxj/4rsA7NC/F0UbjCxZtbyQVh3abO6hNMvOYsDeuzLtfxPoO3RnWrZrve7YHl/dl4evGpW+wtVgXnrsLUY/XX0f6DeoJ8s3uA8ULC2kXaeNr4H2napzhLwWuex3xDBmTp7LgUePaJyCJX0p9RnaXxar+5tHgBBCiy01jjGOijEOjzEO3xZDVIAPJ8yjX++O9OrejqysTL5+zB68OHpyrTYLFhZy4D79AejfrzM5Oc3IL1jDgkUrOGDvnQBonpfFsN17M2OWfyhJW7MJS1fRp00ePVrlkpUR+NpOnXhlzvJabXbp0IJrDuzPOS9MpKCkvIkq1RcxZeUqurfIo2teDs1C4JBunXhnae03v95dWsCR3TsDcHDXjnxYM5zvnaUFHNKtE1kZga55OXRvkceUwlUsLS5lUNtW5GRU/zge1qEtc9c4zcvWYErhKnq0yKNb8+rr4bDunXhrSe3r4a3FBRzVs/p6+Eq3joyr+eP6p29P4OSXx3Dyy2N4ZNZC7ps+n8dnL2JpcSmD27UiJ7P6etizUxvmOKxzq/Dx7BX06dySHh2ak5UZOHZET17+aNG646uKKxh+wTMcdOl/OOjS//DhrIJaIWoIcMyePXjG+VG3OoeeeABX3HUhV9x1IUMP3JV3XxxDjJGZk2aT1yKXth1a12p/yNf354bHr+QPD13OxTefT5eendaFqACfjJ1Gt16da00RoGQbfuxB/Ojmi/nRzRczYN/dmPDq+8QYWTDlU3Ka59Kyfe0Qray4dN28qVWVlcwYM4kOPboA1JpPddq7H9Opd7fGOxF9YUd+8wCuvfdCrr33QoYftCtvvlB9H5g+cTZ5LXNp17H2faCyopJVNetoVFRU8uE7k+nRz9da2lrUp0fqwyGE24G2IYSzgB8Ad6SnrK1DZWUVl4x8kgfvPIvMjAweePx9ps5YwkXnH8lHE+fz4ujJXHn9M9xw9Umcc8aBxAg/u+RhAO7+9zv85Xcn8/ozvyIQePCJD5g8bVGK76itzb03n8+B+w6iY7tWzHjvFq658VHufei1pi5LaVIZ4aq3ZnD3MUPIDIFHpy5mxoq1/Hx4byYsW8Wrcwq4aJ9+NM/K5OYjdgFg4epSzn1xUhNXrrqoinDz5Fn8YcRgMgL8Z/5S5qwu5sz+vZi6cjXvLi3g+flLuGS3AfzzoGGsKq9g5PjqFdrnrC7mtcX53H3gUCqr4OZJM6kCpqxczRuL87lt/92pjJEZRWt4bl71/Ikn9u7GKf260z47mzsOGMr7y1Zww8QZTfgvoA1VRvjzhJncsM8QMgI8N3cJs1et5YcDezGlcDVvLyngubmL+c2wgTxw2J4UlVVw5dgpW3zOyYWreW3Rcu46aA8qY2T6yjU8PWfz82kqOSqrIlc+MJ57f3EgGRmBR96ezfRFRfzi+F2YMGcFr3y05d/x9urfiUUr1jIvf00jVax02HWfQUz43ydc+p3fk52Txfd/feq6Y1f98E9ccdeFKZ/j/VfHO6x/K7bj8F2YMWYSfz/rarJysjn2F6etO3bn+X/gRzdfTFlJKY9ccwcV5RXEqkjv3foz7Jj9AXj1nqdYMmsBhEDbzu05+qenNNWp6AvaY99BjH/3Ey44+fdk52ZxzqXr7wOXnPEnrr33QsrLK7juglFUVlRSVVnFkBEDOPT4fQCY+clc/nzJPaxdVcyHb0/isTtf4Pr7L26q01FTydgKV2DajoT6TGocQjgCOJLqdbVejDH+ty6P6zLo/xy7up0rWjO/qUtQE+tx2Y+bugQ1oV69Mpu6BDWxsnJ/FdjeLXjWUHh7d+/VeakbaZs2a1V9+vFoWzS4XUVTl6AEGN7xayaFW9D/8Du3q1+cp7/8o63qeqjzT7IQQl/gzc/C0xBCXgihT4xxdrqKkyRJkiRJkqQkqM8cqY8AGy4vW1mzT5IkSZIkSZK2afUZW9EsxrhuOdoYY1kIIXtLD5AkSZIkSZJUR2GrGum+3alPj9RlIYTjP9sIIZwA5Dd8SZIkSZIkSZKULPXpkXoucH8I4RaqF5uaB3wvLVVJkiRJkiRJUoLUOUiNMc4E9gkhtKzZXp22qiRJkiRJkiQpQVIGqSGE78YY/xVCuOBz+wGIMd6YptokSZIkSZIkKRHq0iO1Rc3nVuksRJIkSZIkSdquudZUoqUMUmOMt4cQMoGiGOOfG6EmSZIkSZIkSUqUjLo0ijFWAqemuRZJkiRJkiRJSqQ6LzYFvB1CuAV4CFjz2c4Y47gGr0qSJEmSJEmSEqQ+QeoeNZ+v3mBfBA5tuHIkSZIkSZIkKXnqE6SeFGPMT1slkiRJkiRJ0vYsw9WmkizlHKkhhONCCMuAj0MI80MI+zVCXZIkSZIkSZKUGHVZbOp3wIExxh2AbwLXprckSZIkSZIkSUqWugSpFTHGKQAxxveAVuktSZIkSZIkSZKSpS5zpHYOIVywue0Y440NX5YkSZIkSZK0nXGK1ESrS5B6B7V7oX5+W5IkSZIkSZK2aSmD1BjjVXV5ohDCJTFG50+VJEmSJEmStM2pyxypdXVSAz6XJEmSJEmSJCVGQwapzuIgSZIkSZIkaZtUlzlS6yo24HNJkiRJkiRJ25UY7KeYZPZIlSRJkiRJkqQU6hykhhD2T7HvkQapSJIkSZIkSZISpj49Um/e0r4Y4++/fDmSJEmSJEmSlDwp50gNIewL7Ad0CiFcsMGh1kBmugqTJEmSJEmSpKSoy2JT2UDLmratNthfBHwrHUVJkiRJkiRJ250MlyBKspRBaozxdeD1EMI/YoxzGqEmSZIkSZIkSUqUuvRI/UxOCGEU0GfDx8UYD23ooiRJkiRJkiQpSeoTpD4C3AbcCVSmpxxJkiRJkiRJSp76BKkVMca/p60SSZIkSZIkaXvmFKmJllGPts+EEM4LIXQLIbT/7CNtlUmSJEmSJElSQtSnR+oZNZ//b4N9EejXcOVIkiRJkiRJUvLUOUiNMfZNZyGSJEmSJEmSlFR1HtofQmgeQvhNCGFUzXb/EMKx6StNkiRJkiRJkpKhPkP77wHGAvvVbC8AHgGebeiiJEmSJEmSpO1OcLWpJKvPYlM7xhivB8oBYoxrcS0xSZIkSZIkSduB+gSpZSGEPKoXmCKEsCNQmpaqJEmSJEmSJClB6jO0/wrgBaBnCOF+YH/gzHQUJUmSJEmSJElJUucgNcb43xDCOGAfqof0/zzGmJ+2yiRJkiRJkiQpIeo8tD+EsD9QEmN8DmgLXBpC6J22yiRJkiRJkqTtSUbYvj62MvWZI/XvwNoQwu7ABcBM4J9pqUqSJEmSJEmSEqQ+QWpFjDECJwC3xhhvBVqlpyxJkiRJkiRJSo76LDa1KoRwCXA6cGAIIQPISk9ZkiRJkiRJkpQc9emRegpQCvwgxrgY6AH8MS1VSZIkSZIkSdubsJ19bGXqHKTWhKePATk1u/KBJ9JRlCRJkiRJkiQlSZ2D1BDCWcCjwO01u7oDT6ajKEmSJEmSJElKkvoM7f8JsD9QBBBjnA50TkdRkiRJkiRJkpQk9QlSS2OMZZ9thBCaAbHhS5IkSZIkSZKkZGlWj7avhxAuBfJCCEcA5wHPpKcsSZIkSZIkaTsTtsIVmLYj9emRejGwDJgAnAM8D/wmHUVJkiRJkiRJUpLUqUdqCCETmBRj3Bm4I70lSZIkSZIkSVKy1KlHaoyxEpgaQuiV5nokSZIkSZIkKXHqM0dqO2BSCOF9YM1nO2OMxzd4VZIkSZIkSZKUIPUJUi9PWxWSJEmSJEnS9s7FphItZZAaQsgFzgV2onqhqbtijBXpLkySJEmSJEmSkqIuc6TeCwynOkQ9GrghrRVJkiRJkiRJUsLUZWj/LjHGXQFCCHcB76e3JEmSJEmSJElKlroEqeWffRFjrAjO1SBJkiRJkiQ1vLqMHVeTqUuQunsIoajm6wDk1WwHIMYYW6etOkmSJEmSJElKgJRBaowxszEK+X/27jw+rrre//j7O9mbvdmarmm6QTfa0kLZWgoFBAVF7lXBy6aCIlz1AoJsLVDEq4iKigg/RMqi7CJLvSCllFJaoBt039IlS7Pv20wm8/39MaFJup2EZjInzev5eMyDzjnfM/kc5uTMmXe+5/sFAAAAAAAAALfqSo/Uo3bcA1/rjR8DFysqbA13CQizgp8/Eu4SEEYFkiY9eH24y0CYVbyaH+4SEEZxc4eEuwSE2cu7/eEuAWEWsAwT198tLooNdwlwgadmh7sC4Itj5AUAQMgRooIQFQAAAEBf1ys9UgEAAAAAAAA4YJJ3V6NHKgAAAAAAAAA4IEgFAAAAAAAAAAcEqQAAAAAAAADggCAVAAAAAAAAABww2RQAAAAAAADgBsw15Wr0SAUAAAAAAAAABwSpAAAAAAAAAOCAIBUAAAAAAAAAHDBGKgAAAAAAAOAC1sMgqW5Gj1QAAAAAAAAAcECQCgAAAAAAAAAOCFIBAAAAAAAAwAFBKgAAAAAAAAA4YLIpAAAAAAAAwA0Mk025GT1SAQAAAAAAAMABQSoAAAAAAAAAOCBIBQAAAAAAAAAHBKkAAAAAAAAA4IDJpgAAAAAAAAA3YK4pV6NHKgAAAAAAAAA4IEgFAAAAAAAAAAcEqQAAAAAAAADggDFSAQAAAAAAADfwMEiqm9EjFQAAAAAAAAAcEKQCAAAAAAAAgAOCVAAAAAAAAABwQJAKAAAAAAAAAA6YbAoAAAAAAABwA8NkU25Gj1QAAAAAAAAAcECQCgAAAAAAAAAOCFIBAAAAAAAAuJIx5kvGmK3GmB3GmJ8dYv1VxpgyY8y6tsf3Oqy70hizve1x5dHWwhipAAAAAAAAAFzHGBMh6WFJ50gqkPSJMeY1a+2mA5o+b6294YBtB0qaL2m6JCtpddu2VV+0HnqkAgAAAAAAAG5g+tnD2UmSdlhr86y1PknPSfpql7aUzpP0b2ttZVt4+m9JX+ritodEkAoAAAAAAADAjYZIyu/wvKBt2YEuMcZ8Zox5yRgzrJvbdhlBKgAAAAAAAIBeZ4y51hizqsPj2i/wMq9LyrHWTlaw1+nCnq2yHWOkAgAAAAAAAOh11trHJD12hCaFkoZ1eD60bVnH16jo8PRxSb/qsO2ZB2z73hcsVRJBKgAAAAAAAOAOnq4NHNqPfCJpjDFmpILB6LckXdaxgTEm21q7r+3pRZI2t/37LUn3G2NS256fK+m2oymGIBUAAAAAAACA61hr/caYGxQMRSMkPWGt3WiMuVfSKmvta5J+ZIy5SJJfUqWkq9q2rTTGLFAwjJWke621lUdTD0EqAAAAAAAAAFey1i6StOiAZfM6/Ps2HaanqbX2CUlP9FQtTDYFAAAAAAAAAA4IUgEAAAAAAADAAbf2AwAAAAAAAG7AZFOuRo9UAAAAAAAAAHBAkAoAAAAAAAAADghSAQAAAAAAAMABQSoAAAAAAAAAOGCyKQAAAAAAAMAFLHNNuRo9UgEAAAAAAADAAUEqAAAAAAAAADggSAUAAAAAAAAAB4yRCgAAAAAAALiBh0FS3YweqQAAAAAAAADggCAVAAAAAAAAABwQpAIAAAAAAACAA4JUAAAAAAAAAHDAZFMAAAAAAACAGxgmm3IzeqQCAAAAAAAAgAOCVAAAAAAAAABwQJAKAAAAAAAAAA4YI/ULOCkjRf89IVceI725t0R/21nYaX2Ux+j2KWM1NjletT6/7lmzVcVNXk1PT9a1x+UoymPUErB6ZPNura2okSTNyU7X5WOGymOMVpRU6tEte8KxazhKZwxL1Z2njlKEMXphS7EeW5ffaf3Vk4boG8cPkj9gVdncotve26aiem+YqkVv+PMD39f5Z09VWUWtpp9zS7jLQQ+amZWim6bmymOM/plXoqe2FnRaH+UxuvuksTouNUE1Xr/uWLlF+xq9ijBGd04frXGpCYowRov2lGrhluC23xw9WF/LzZKR9OquEj23vSgMe4YvYtbkbN11+TRFeIyef2+nHn19c6f1l8waqVsvnaKSqiZJ0tNvb9ML7+Xp+BEpuvfqGUqIi1IgYPWnf27Umyv3hmMXcJROH5Kqn50cvAZ4eVuxHl/f+RrgxKxk/ezkXI1NTdBP39ust/eUS5JOGpSsW08atb/dyOQBunnpZr27t6JX68fRs9Zq87MvqOzTjYqIjtaka65Qcs7wg9pte+mfKlz+kVoaGnXuY7/bv3zzsy+qYss2SVKr1ydfXZ3OeeQ3vVY/jp61VluefUFln20IHgPfu1JJhzgGtr/0qoo+DB4Dcx99qNO64o9Xacerb0gyShw+VCf84Lu9VD16mrVWe55/XtXr18sTHa1RV12l+BEjDmqX/49/qHzlSvkbGzXjD38IQ6UAuoMgtZs8kn4yMVc3fbRRZU0+PXrGCVpeUqk99U3723x5WJbqWvz69pI1Omtwur5/fI7uWbNVNT6/bvtksyq8Po1MHKAHTh6v/3hnlZKiInXd+Bxds2xdsM0JYzQtLVlr2kJW9A0eI9192mhd9eZ6FTd49fLXp+rd3RXaUd24v82minpd/MpaNfsDumx8tm6ZOVI/eWdLGKtGqD394lL9eeFbevy3Pwx3KehBHkm3TBulG97foNJGnxbOnaJlRRXaVdf+WXDRyCzV+fy65F+rdc6wdN0wOUd3rNyquUPTFeXx6LK31yomwqPnz5umt/eWKS4yQl/LzdJViz+VPxDQQ2dM1AdFlSpoaA7fjqJLPMbo7qtO1JW/WKLiyib9Y8G5WrymUDsKazu1e3PlXt2zcHWnZU3eVv30kRXaXVKvzJQ4/fO+8/T+Z/tU19jSm7uAo+Qx0h0zR+uat9arpNGr5y+cqiV7K7Szpv0aYF9Ds+5Ytk1XTRzaaduPi2t0yWtrJEnJ0ZH613/M0IeFVb1aP3pG2Wcb1VBcqlm/ukfVO3dp48K/69T5tx7ULmPKJA2fe6bev2V+p+XHf/s/9/9797+XqHZP/oGbwuXKP9ugxpJSnfHLe1Wzc5c2PfU3zZz3s4PaZUyZrOFz52jZrfM6LW8oLlHeG2/p5Dt+qqj4eHlraw/aFn1HzYYNai4p0Qn33af6Xbu069lnNfH22w9ql3LCCcqaM0ef3nVXGKqEK3mYbMrNuLW/m45PSVRhQ7P2NXrlt1bvFpbp9KyBndqcljVQb+WXSpKW7ivXtPRkSdL22gZVeH2SpF11jYrxeBTlMRo8IFYFDU2q8fklSavLqzU7O60X9wo9YXJmovbUNim/rlktAas3d5Tp7JzO7+NHRTVq9gckSetKajUoPiYcpaIXLf94iyqr68NdBnrYhIGJKqhvVlFD8LPg7fwyzRrS+fd99uA0vbk7+Fnwzjm//wAAIABJREFUbkG5ZmSmSJKsrOIiIxRhpNgIj/wBq4aWVo1MitPGyjp5WwNqtdKashrNGcpnQV9wwqiB2lNSr/yyBrW0BvTGyr2ae+JQ5w0l7S6u0+6S4DmitLpJFbXNSkvks6GvmZSeqPy6JhXUB68BFuWVac7wzr+/RfVebatqkLX2sK9zbk66lhVUqbk1EOqSEQKlaz7VkNNmyhij1NG58jc2qrn64I4RqaNzFZuSfMTX2rdylQbPnBGqUhEipWs/0+C2YyBldK5aGpvkPcQxkDI6VzGHOAYKln6g4WfPVlR8vCQpJikp5DUjdKrWrVP6KafIGKPE3Fy1NjXJV119ULvE3FxFp6SEoUIAXwRBajelx0WrtNm3/3lZs0/pcZ2/8KTHRqu0OXi7dquVGlr8So7q3Pl3dnaattU0qCVgVdDYpGHxcRoUF6MII50+aKAy4/gS1dcMGhCjfR1u0y9u8CorPvqw7f/juEF6fy89ToC+KCMuWiWN7b/vpY1eZcRFH9ymqf2zoL7Fr+ToSC0uqFCTv1WLLjxZr315hp7ZWqDaFr921jRqSnqykqMjFRPh0WnZqcris6BPyBo4QPsq2nseFlc2Kis17qB2X5oxTG/+4nz98cenKXvggIPWT84dqKhIj/aU8seXviZrQIz2NbSfE0oaj3wNcDjn52ZqUV5pT5aGXtRcVa3YtNT9z2MHpspbdXBo4qSpvEJNZeVKGz+uJ8tDL/BWVSt2YIdjIDVFzd04BhqLS9VQXKKP7vuVVt77S5V9tjEUZaKX+KqrFZPafjxEp6YeMkgF0Ld0+dZ+Y0yGpGsk5XTczlr7nZ4v69iWkxCn7x83Qjd/tEmSVN/Sqt+u36n508bJympDZZ0Gx8eGuUqE0kVjMjUpI1Hffu3TcJcCoJdNGJiggLW64PWPlRQdqcfmTNLHpdXaXdekp7YU6PezJqrZ36pt1Q1qPULPNfQti9cU6vUP98jnD+jSs0bpgR/M1H/d/+7+9RkpsXrwulP000dXire9f0qPi9aY1AFazm39/V7RR6s0aMY0GQ99XvobGwiosaRUM352k5qrqvTJLx7UqQvuUlT8wX98AwCER3fGSP2npGWS3pHU6tTYGHOtpGslacz1P1X2l776hQp0m/ImnzJj23sYZMRGq7yp82RB5c0+ZcbGqKzZpwgjxUdFqqbFv7/9fdOP1/3rtquosX3cuw9Lq/RhafDC+cLhWQqIb1F9TXGjV9kJ7b3HBsXHqKTBd1C7U4ek6IdTh+uy1z6VL8D7DPRFZU0+ZQ1o/33PHBCjsibfwW3iYlTaFPwsSIiKVI3Pr/OGZ2hFcZVarVWVt0WfltdpfGqiihq8em13iV7bXSJJum7iCJU2MRldX1BS2ajstPYvuYMGDtg/qdTnquvbj4/nl+Tp1kun7H+eEBepx2+erQdf/EzrdjDBUF9U0uhVdofherIGHPoa4Ei+NDJdi/dUyE+S3qfseec95S9dLklKHjlCzRXtQXhzZZViUrt/u+6+las04Ypv9ViNCK2977yngqUfSJKSRo5Qc2WHY6CqWrHdOAZiUlOUMmqkPJERGpCRrgFZmWosKVVybk5Pl40QKV6yRGXLlkmS4nNy5K2qUmLbOl9VFbfwo2v4O5qrdeftGWCtvdVa+4K19uXPH4drbK19zFo73Vo7/VgJUSVpS02dhrbdhh9pjM4akqHlJZWd2iwvqdR5wzIlSbOz07W2PDguTkJkhP73pPF6dMtubaiq67RNSnRUsE1UhL46YpDe2FvSC3uDnrS+tE45yXEamhirKI/Rl0dnaPGezl+Ix6fFa8EZY/T9/9ugymYmEgH6qk1VdRqWEKfBA4KfBecOy9Cyos6fBe8XVerLOcHPgrOGpmtVafBWrpJGr6a3jZcaG+HRxLRE7a4L3haeGhP8LMiKi9GcIWl6a29Zb+0SjsJneZXKGZSooRnxiorw6Cszh2vx6oJObTJS2u80mXviEO0oCk4gEhXh0SM/OUP/+GC3/u9jJpbpqzaU12l4UpyGJASvAS7IzdCS/O6F4heM5Lb+vmjE3DN1+oI7dPqCO5Q17QQVLl8pa62qduQpMi7OcSzUA9UXFcvf2KiU0bkhqhg9bfjcM3Xqgjt16oI7lTVtiorajoHqHXmKjIs95Fioh5M5bYoqt2yTJPnq6tVYUqq4zPRQlY4QGDRnjibNm6dJ8+YpdcoUla9YIWut6vLyFBEXR5AKHAO60yP1DWPMBdbaRSGrpg9otdLvNubp1ydPkMdIi/JLtbu+Sd8ZO1xbaur1YUmlFuWX6I4pY/XsnGmqa/HrnjVbJUkX52RryIBYXTlmmK4cM0ySdPNHm1Tta9GPJozUqKTgoOILt+czS3Mf1Gqlez7YoScumKgIY/TS1mLtqGrUj6eP0PqyOr27p1K3zMzVgKgI/eGc8ZKCE0/84C3GPjqWLfzDf+uMU45Xemqidnz0Ry34zUta+Px74S4LR6nVSg+s3anfz5ooj5Fe31WivNpGXTthuDZX1mvZvkq9tqtY95w0Ti+ff6JqfX7dsXKLJOnFHfs0b8ZYPXfuVMkYvbGrRDvaZvb+5SnHKSkmSq0BqwfW7lR9i+MNIHCB1oDVPU+u0pO3nimPx+ilpXnaXlirn1wySet3VWrxmkJded44nT1tiFpbA6pp8OmWP6+UJF0wc7hmHJeplMQYXTJrpCTplkdXavMexlDrS1qt9POVO/TYuRPlMUb/2F6sndWNumHqCG0sr9OS/EpNTE/QQ2dNUFJ0pM4clqbrp47QV19dLUkanBCjQfEx+qT44Elp0HdknDBRZZ9t0NKfzlNETLQmf++K/es+uOvnOn3BHZKkLc+/oqIVn6jV59O7P7lNw2afpjEXf0WStO+jVco+ebqMYcbmvii97RhYdstdioiJ1sTvXrl/3Yd33adTF9wpSdr6/MvatzJ4DLz3Pz/T0FmnafTFFyp90nhVbNykD26/W8bj0dhvfF3RCQnh2h0cpZRJk1S9YYM+veMOeaKjlXvVVfvXrb/3Xk2aN0+StPell1T+8ccK+Hxac8styjz9dA296KIwVQ3AiTnSzKGdGhpTJylekq/tYSRZa63jVIKz31jOPUr9XFEhYUB/V/DzR8JdAsJo0oPXh7sEhFnFq/S27O9i5w4JdwkIs3OO94e7BIRZwBIQ93fVPu5ZhvTU7NmcDI4g9/pX+lWGlvfw1/vU8dDlHqnW2kTnVgAAAAAAAABw7OlykGqC95d8W9JIa+0CY8wwSdnW2o9DVh0AAAAAAADQXzC8i6t1p1/9nySdIumytuf1kh7u8YoAAAAAAAAAwGW6M9nUydbaacaYtZJkra0yxkSHqC4AAAAAAAAAcI3u9EhtMcZESLKSZIzJkBQISVUAAAAAAAAA4CLdCVJ/L+kfkjKNMT+X9IGk+0NSFQAAAAAAAAC4SJdv7bfWPmuMWS3pbElG0testZtDVhkAAAAAAADQn3iYbMrNuhykGmN+L+k5ay0TTAEAAAAAAADoV7pza/9qSXcaY3YaY35tjJkeqqIAAAAAAAAAwE26HKRaaxdaay+QNEPSVkm/NMZsD1llAAAAAAAAAOASXb61v4PRko6TNEISY6QCAAAAAAAAPcAaxkh1sy73SDXG/KqtB+q9kjZImm6tvTBklQEAAAAAAACAS3SnR+pOSadYa8tDVQwAAAAAAAAAuJFjkGqMOc5au0XSJ5KGG2OGd1xvrV0TquIAAAAAAAAAwA260iP1RknXSnrwEOuspLN6tCIAAAAAAAAAcBnHINVae60xxiPpTmvt8l6oCQAAAAAAAOh/ujybEcKhS2+PtTYg6Y8hrgUAAAAAAAAAXKk7OfdiY8wlxhgTsmoAAAAAAAAAwIW6E6R+X9KLknzGmFpjTJ0xpjZEdQEAAAAAAACAa3RlsilJkrU2MZSFAAAAAAAAAIBbdSlINcZESjpf0nFtizZJesta6w9VYQAAAAAAAEC/4mFETTdzvLXfGDNE0kZJN0kaLGmIpFskbTTGDA5teQAAAAAAAAAQfl3pkfpzSY9Ya3/XcaEx5keSfiHpylAUBgAAAAAAAABu0ZUgdaa19qoDF1prf2+M2drzJQEAAAAAAACAu3QlSG06wrrGnioEAAAAAAAA6NcMY6S6WVeC1GRjzNcPsdxISurhegAAAAAAAADAdboSpC6VdOFh1r3fg7UAAAAAAAAAgCs5BqnW2qu78kLGmCuttQuPviQAAAAAAAAAcBdPD77Wj3vwtQAAAAAAAADANbpya39XMRouAAAAAAAA8EV5iNfcrCd7pNoefC0AAAAAAAAAcI2eDFKJzAEAAAAAAAAck3oySF3eg68FAAAAAAAAAK7R5SDVGPNjY0ySCfqLMWaNMebcz9dba28ITYkAAAAAAAAAEF7d6ZH6HWttraRzJaVKulzS/4akKgAAAAAAAKC/Mf3s0cd0J0j9fPcukPS0tXaj+uQuAwAAAAAAAED3dCdIXW2MeVvBIPUtY0yipEBoygIAAAAAAAAA94jsRtvvSpoiKc9a22iMSZN0dWjKAgAAAAAAAAD36HKQaq0NGGP8kmYZYzpu91nPlwUAAAAAAAD0L9bDKJpu1uUg1RjzhKTJkjaq/ZZ+K+mVENQFAAAAAAAAAK7RnVv7Z1prx4esEgAAAAAAAABwqe5MNrXCGEOQCgAAAAAAAKDf6U6P1KcUDFOLJXklGUnWWjs5JJUBAAAAAAAAgEt0J0j9i6TLJa1X+xipAAAAAAAAAHoCk025WneC1DJr7WshqwQAAAAAAAAAXKo7QepaY8zfJL2u4K39kiRr7Ss9XhUAAAAAAAAAuEh3gtQ4BQPUczsss5IIUgEAAAAAAAAc0xyDVGPMMGttvrX26kOs+0poygIAAAAAAAAA9/B0oc2/jTE5By40xlwt6aGeLggAAAAAAADol4zpX48+pitB6o2S3jbGjPl8gTHmtrbls0NVGAAAAAAAAAC4heOt/dbaRcYYr6R/GWO+Jul7kk6SNMtaWxXqAgEAAAAAAAAg3LrSI1XW2sWSrpb0nqRcSWcRogIAAAAAAADoL7oy2VSdJCvJSIqRdLakUmOMkWSttUmhLREAAAAAAADoB7rU5RHh0pVb+xN7oxAAAAAAAAAAcCtybgAAAAAAAABwQJAKAAAAAAAAAA4IUgEAAAAAAADAgeMYqQAAAAAAAAB6gTHhrgBHQI9UAAAAAAAAAHBAkAoAAAAAAAAADghSAQAAAAAAAMABQSoAAAAAAAAAOOiVyaaGJPh748fAxSKHM69Zf5f64PXhLgFhtv6mh8NdAsIoZ/514S4BYXbccCZO6O9WlUSHuwSEWUa8DXcJCLM9peGuAOgDPFwzuRk9UgEAIUeICgAAAADo6whSAQAAAAAAAMABQSoAAAAAAAAAOGDgSgAAAAAAAMANGCPV1eiRCgAAAAAAAAAOCFIBAAAAAAAAwAFBKgAAAAAAAAA4IEgFAAAAAAAAAAdMNgUAAAAAAAC4gDVMNuVm9EgFAAAAAAAAAAcEqQAAAAAAAADggCAVAAAAAAAAABwQpAIAAAAAAACAAyabAgAAAAAAANyALo+uxtsDAAAAAAAAAA4IUgEAAAAAAADAAUEqAAAAAAAAADhgjFQAAAAAAADADYwJdwU4AnqkAgAAAAAAAIADglQAAAAAAAAAcECQCgAAAAAAAAAOCFIBAAAAAAAAwAGTTQEAAAAAAABu4GGyKTejRyoAAAAAAAAAOCBIBQAAAAAAAAAHBKkAAAAAAAAA4IAgFQAAAAAAAAAcMNkUAAAAAAAA4AZMNuVq9EgFAAAAAAAAAAcEqQAAAAAAAADggCAVAAAAAAAAABwwRioAAAAAAADgBgyR6mr0SAUAAAAAAAAABwSpAAAAAAAAAOCAIBUAAAAAAAAAHBCkAgAAAAAAAIADJpsCAAAAAAAAXMB6mG3KzeiRCgAAAAAAAAAOCFIBAAAAAAAAwAFBKgAAAAAAAAA4IEgFAAAAAAAAAAdMNgUAAAAAAAC4gWGyKTejRyoAAAAAAAAAOCBIBQAAAAAAAAAHBKkAAAAAAAAA4IAxUgEAAAAAAAA38DBGqpvRIxUAAAAAAAAAHBCkAgAAAAAAAIADglQAAAAAAAAAcECQCgAAAAAAAAAOCFIBAAAAAAAANzD97NGV/yXGfMkYs9UYs8MY87NDrL/RGLPJGPOZMWaxMWZEh3Wtxph1bY/XuvYTDy/yaF8AAAAAAAAAAHqaMSZC0sOSzpFUIOkTY8xr1tpNHZqtlTTdWttojLlO0q8kfbNtXZO1dkpP1UOPVAAAAAAAAABudJKkHdbaPGutT9Jzkr7asYG1dom1trHt6UpJQ0NVDEEqAAAAAAAAADcaIim/w/OCtmWH811J/+rwPNYYs8oYs9IY87WjLYZb+wEAAAAAAAD0OmPMtZKu7bDoMWvtY1/wtf5L0nRJszssHmGtLTTG5Ep61xiz3lq784vWS5AKAAAAAAAAuICnn9073haaHik4LZQ0rMPzoW3LOjHGzJV0h6TZ1lpvh9cvbPtvnjHmPUlTJX3hILWfvT0AAAAAAAAA+ohPJI0xxow0xkRL+pak1zo2MMZMlfSopIustaUdlqcaY2La/p0u6TRJHSep6jZ6pAIAAAAAAABwHWut3xhzg6S3JEVIesJau9EYc6+kVdba1yQ9IClB0ovGGEnaa629SNLxkh41xgQU7Ez6v9ZaglQAAAAAAAAAxx5r7SJJiw5YNq/Dv+ceZrsPJU3qyVq4tR8AAAAAAAAAHNAjFQAAAAAAAHCB4J3pcCuC1C/AWqv8559X7Yb18kRHK+eqqzRg+IiD2jXs2aPdT/5VtqVFSRMnadg3vyljjBrz87X32WcUaGmR8URo+GWXKX7kSLU2NWrXX56Qr6pStrVVWeecq/TTTgvDHuJQZqSn6Prjc+Ux0qKCEj2X13mSuCiP0a2Tx2psUrxqW/xasG6rSpqCE8VdmjtE5w/NUsBKf9ycp1Xl1ZKkS3IG64KhWbKy2lXXqF+t366WgNVXhw/SJTmDNSQ+The/85FqW/y9vr84splZKbppaq48xuifeSV6amtBp/VRHqO7Txqr41ITVOP1646VW7Sv0asIY3Tn9NEal5qgCGO0aE+pFm4JbvvN0YP1tdwsGUmv7irRc9uLwrBn6Gl/fuD7Ov/sqSqrqNX0c24JdzkIkdOHpOr2maPk8Ri9tLVYj3+W32n99EHJuu3kXI0dmKCblmzW27vL96+7ecZIzR42UMYYfVhYpftXfuFJRNHLrLXa9+LfVb9xvUxUtIZe8R3FHeKasGnvbhU89VfZFp8SJkxS9n9eKmOMSt74p6qWL1NkYqIkKeuii5U4cbIad+ep6G9Pf/5DlPnli5Q0ZVpv7hq66OTMFP14Uq48Mnpjb4me2X7w9cCd08ZqXHKCalv8mvfJFhU3eTUoLkbPnj1Ne+ubJEkbK+v068+Cv/uRxujGyaM0NT1ZAWv12OY9Wrqvotf3Dd1nrVXJi39X3cbg98TBlx/+nFD09F8V8PmUOGGSstrOCaVv/lPVy5cpIiF4TshsOyeg7zg1O1U3n5irCGP0j53FenJT53PCtIwk3XTiKI1Jiddty7docX7wemBsSrxuP2m04iMjFLDSXzbu1dt7yw/1IwCEGUHqF1C7YYO8pSWasOA+NezapT3PPqvjb7v9oHZ7//asRlx+heJHjtSOP/xetRs3KHniJBW8/JKyv/IVJU+cpJr161Xwyssad9PNKl3ynmKzszX6hhvUUlenjfPu0sCTT5Ynkrcp3DySfjQhV7d8vFFlzT796dQTtKK0UnvaLn4l6fyhWapv8euK99doTna6rhmXo/vWbdWIhDjNyc7Qdz9Yq7SYaD1w0gRduXSNBsZE6+IR2frOsrXyBQK6a8o4nZWdobcKS7Wxuk4rP9mo35w0MXw7jcPySLpl2ijd8P4GlTb6tHDuFC0rqtCuuvbj4aKRWarz+XXJv1brnGHpumFyju5YuVVzh6YryuPRZW+vVUyER8+fN01v7y1TXGSEvpabpasWfyp/IKCHzpioD4oqVdDQHL4dRY94+sWl+vPCt/T4b38Y7lIQIh4j3XXqaH33/9arpMGrFy6aqiV7K7SzunF/m6L6Zt32/jZ9Z9LQTttOyUzS1KwkffUfqyVJz35limYMStYnxTW9ug/4Yuo3rpevtFRj7r5fTbvzVPTcMxp1yx0HtSv6+zMa8u0rFJeTqz0PP6T6TRuUOCE4XFf6Weco/ZzzOrWPHTxEo269UyYiQi011drx83uUOOkEmYiIXtkvdI1H0o2TR+l/Ptyg0iafHp89RR8UV2h3h+uBrwwPXg98a/FqnT0kXddNyNH8VVslSYUNzbr6vXUHve4VY4epyuvTpYtXy0hKiua7QF9Rv3G9vGWlGt12Ttj33DPKPcQ5Yd9zzyj7suA5Ye+fOp8TBp51jtLnnnfQNnA/j5FunT5KP3x3g0qavHrmvClaWlCpXbXt1wP7Gr26e+VWXX585+uB5taA7lqxVfl1zUqPi9azX5qqD/dVqb6ltbd3A4ADxkj9Aqo/Xae0mafIGKOE3Fy1NjWppaa6U5uWmmq1NjUpITdXxhilzTxF1euCF0rGGLU2BcOR1qYmRSWnqG2FAt5mWWsV8HoVGR8v4+EtcoPjUhJV2NCsfU1e+a3Vkn1lOjVzYKc2p2YO1NuFpZKkpcXlmpaWvH/5kn1laglYFTd5VdjQrONSgn9ljjBGMREeeYwUG+FRudcnSdpR27C/NyvcZ8LARBXUN6uoIXg8vJ1fpllD0jq1mT04TW/uDh4P7xaUa0Zm8PfcyiouMkIRbe+5P2DV0NKqkUlx2lhZJ29rQK1WWlNWozlD0w762eh7ln+8RZXV9eEuAyE0OSNRe2ubVFDXrJaA1aK8Mp01vPPvb1G9V9uqGhSw9oCtrWIiPIryeBTt8SjSGFU0+XqveByV2s/WKeXk4DXhgJGj1NrYeOhrwuZmDRg5SsYYpZx8imo/XXvE1/VEx+wPTW1Li8Qtfq50fGqiChqaVdQYvB54p7BMpw/q/Lt/enaa/pUfvB54r6hcJ6anOL7ul0dk6em2nq1WUo2PO5P6iroDzgmBpkOfEwIHnBPqHM4J6BsmpgW/IxQ2NMsfsHprT5nOHNr5O+O+Bq+2VzcqcMDlwN66JuXXBTOC8iafqpp9So2N6q3SAXRDl/68aYyJkPQja+1vQ1xPn9BSXa3ogan7n0enpMpXVd0eiEryVVUrOrW9TVRqqlqqgx+iQ7/xTW1/6HcqePklyVqNu+VWSVLmnDna8fAf9dktP1XA61XuNdcQpLpEemy0yprbv9iWNft0fFsY2rFNaXMw/AxYqcHvV1JUpNJjY7S5um5/u/Jmn9Jjo7Wpuk4v7irU38+cLm8goFXl1Vpd3vlCC+6UERetksb2oLu00asJaYkHt2kLw1utVN/iV3J0pBYXVGjW4DQtuvBkxUZ49Nt1eapt8WtnTaOum5ij5OhINbcGdFp2qjZXEr4BfUHmgBgVN7SfE0oavZqckXiELdqtK63TR/uq9f6lM2WM9OymIuXVNDlvCFfwV1crKrX9S3JUampwWYdrQn91taJSUg9q87mKpe+q6qMPFTciR9mXfEMRA+IlSY278lT4zJNqqazQ0Cu/S29UF8qIjVZphz98lzV5NT418bBtWtuuD5PbephmD4jVE7OnqMHfqv+3eY8+q6xVQmTwff7ecSM0NT1ZRQ3N+s36narytvTSXuFo+GuqFZXSfk6ITHE+J0SmpMrfIWytWvquaj76UHHDc5TV4ZwA98uI63w9UNro08T0rl0PdDQhLUFRHo8K6rgzrb9ijFR361KQaq1tNcZcKokgtQeULV2qYd/4hlKnnajKVau056mFGvs/N6p240YNGDZMY2+8Sd6yMm3/3W81fvQYRcTFhbtkhEBCZIROzRqoby9dpfqWVs2fOk5zB2fonaKycJeGEJowMEEBa3XB6x8rKTpSj82ZpI9Lq7W7rklPbSnQ72dNVLO/VduqG9R6UM81AMea4YmxGpUyQHOeWylJ+sv5k/VBQZJWl9SGuTL0hrRZZyrzggslSaWvv6p9L7+goZdfLUkaMDJXY+66V837ilT41BNKmDBJnih6Jx0rKrw+XfL2J6pt8WtccrzuP3m8Ln93jSI8RllxMdpQWas/btylb44arOsnjNR9a7aFu2T0goFnnKmM84PnhLI3XlXJyy9ocNs5Af1DemyUFpwyTvNXbBPfBAB36k53x+XGmD8aY84wxkz7/HG4xsaYa40xq4wxq3a8/noPlBpepUuWaNOCe7Vpwb2KSk6Wr7Jq/zpfdZWiUzvfphOdmiJfVXublqoqRaUE21Ss+FApU4P/61JPPFENu3dLkso/XK6UqdNkjFFsZqZi0tPVXFwc4j1DV5Q3+5QRG73/eUZstMqbvQe1yYyNkRQcHyc+MlK1LX6VN3s7bZseG63yZp+mpaeouNGrGp9frdZqWXGFxqd0/y+W6H1lTT5lDYjZ/zxzQIzKDrgVt6zJp6y4YJsIIyVERarG59d5wzO0orhKrdaqytuiT8vr9vdeeW13ia58Z52+/9561fr8+yegAOBupY1eDYpvPydkDYhRSUPXbs+fm5OuT0vr1OgPqNEf0LL8Sk3JTApVqegBFUvf1Y7779GO++9RZHKyWqoq969rqapSZErna8LIlBS1VFcdsk1kUrKMxyPj8Sj19Flq2r3roJ8Xmz1YnphYeYsKD1qH8Cpr9ikzrv13PyMuptMdTAe2iWi7Pqzx+dUSsPsnE91a06CihmYNS4hTjc+vJn/r/smllhSWa1wyPRLdrHLpu9p5/z3aef89ikxKVkt1+znBX+18TvBXVyky+eBzQspps9S05+BzAtyrrKnz9UDmgGiVNnZ9uLb4yAg9dOZEPfzpHq05WOyKAAAgAElEQVSvqHPeAEBYdCdInSJpgqR7JT3Y9vj14Rpbax+z1k631k4ffeGFR1elC2TOmaPxd83T+LvmKWXKFFWsXCFrrerz8hQRF9fpdg1JikpOUURcnOrz8mStVcXKFUo5YYokKTolRfXbgn9VrtuyRbGZmcHlA9NUt2WzJKmltlbNJSWKyUjvxb3E4WypqdOQ+DgNiotRpDGak52hD0srO7VZUVqpc4cE38vZg9K1tiI4UciHpZWak52hKI/RoLgYDYmP05bqOpU2eXV8SqJi2oZvmJaWor0NBGd9waaqOg1LiNPgAcHj4dxhGVpW1Pl4eL+oUl/OCR4PZw1N16rS4C1bJY1eTW8bLzU2wqOJaYnaXRccgD41JtjTKCsuRnOGpOmtvfROBvqC9WV1GpEUpyEJsYryGF2Qm6Ele7s2w/a+eq9mDEpWhAnO1D09O7nTJFVwn7TZZ2n07fM1+vb5Spo8VdUfBa8JG3ftPPw1YWysGnftlLVW1R+tUNLk4DVhx7ETa9etUezgIZIkX3mZbGtwghFfRYW8JfsUlca42W6zpbpOw+LjlN12PTB3SIaWF3e+HlheXKnzhwWvB84cnK41bcM4pURH7v8iNnhAjIbGx6qobYLJ5cWVmpoeHGv/xIyUTpNXwX0Gzj5Lo26fr1G3z1fiCZ3PCZ7DnBM8B5wTEg9xTqj7dI1i2s4J6Bs2VtRpWGKsBsfHKNJjdN6IDC0trHTeUFKkx+jBWeP15q4SLc4vD3GlAI6Gsb1w6+hl7y09pnqlW2uV//e/q2bjBnmio5Vz5VWKz8mRJG1acK/G3zVPktSwe7d2L3xSAZ9PyRMnati3LpUxRvU7tiv/+edlAwGZyEgNv+zbih8xQr7qau1+8q9qqQkGcIPO+5LSZs4M1272qJKmvj/b6EkZqbr++JHyGOlfBaX6284CXTVmuLbW1GtFaaWiPEa3TR6r0Unxqmvx6751W7WvbUysy0YN1flDM9UakP60OU8ft11EXzl6mM7MTlertdpR26AHN+xQS8Dq4hHZ+mbuEA2MjlaVr0Ufl1XpwQ07wrn7R62u/pg6DejUQam6cUquPEZ6fVeJ/rqlQNdOGK7NlfVatq9S0R6je04ap7Gp8ar1+XXHyi0qavAqLsKjeTPGamRSnGSM3thVome2BXsZPXbmJCXFRKk1YPW7T/P0SemxM2v3+pseDncJYbPwD/+tM045XumpiSotr9GC37ykhc+/F+6yel3O/OvCXUJIzRqaqttmjpLHGL2yrViPfpqv/542QhvK67Rkb6UmpifoD3MnKCk6Ur7WgMqbfLrwldXyGGneqWM0fVCyrLX6oLBKv/woL9y7ExLjc469Ab+stdr3/N9Utyl4TTj08qsVNyJHkrTj/ns0+vb5kqSmPbtV8NQTCrS0KHHCRGV/4zIZY5T/5ONqLsiXJEWnpWvwZZcrKjlFVR+tUPnb/wqOi2qMMs+/UElTpoZrN3tMSf2xdwzMzEzVjycFrwfe3Fuip7YV6LvHDdeW6notLw5eD9w1bZzGJMertsWvu1dtUVGjV7Oz0/S944bLb60CVnpiy14tLwkGLllxMbpr2lglREWq2teiX6zdfsxMQpoRf2xdDx7IWqviF/6m+rZzwuD/aj8n7Lz/Ho3qcE4oejp4TkgYP1GD2s4JhU8+rubC4DkhKi1d2ZdeflAQ29ftKQ13BaF12uBU3TwtVx5j9Fpeif6yMV8/mDRCmyrr9H5hpcYPTNCDs8YrKTpS3taAKpp8+s9Fa3RBTobmzxyrvJr2P6bOX7FN26obwrg3obPmsjOOvQ+EHpT7p2MrQ3OS98PZfep46HKQaozJknS/pMHW2vONMeMlnWKt/YvTtsdakIruOxaCVBydYy1IRff05yAVQcd6kApnx2KQiu45FoNUdM+xHqTC2bEepKJrCFKPbNQj7/erk+XO62b1qeOhO7f2PynpLUmD255vk/STni4IAAAAAAAAANymO0FqurX2BUkBSbLW+iW1hqQqAAAAAAAAAHCR7gSpDcaYNElWkowxMyUdOwP4AQAAAAAAAMBhdGfgyhslvSZplDFmuaQMSf8RkqoAAAAAAAAAwEW6HKRaa9cYY2ZLGifJSNpqrW0JWWUAAAAAAABAP2L61NRL/U93p1I/SVJO23bTjDGy1j7V41UBAAAAAAAAgIt0OUg1xjwtaZSkdWqfZMpKIkgFAAAAAAAAcEzrTo/U6ZLGW2ttqIoBAAAAAAAAADfqTpC6QdIgSftCVAsAAAAAAADQbzFGqrt1J0hNl7TJGPOxJO/nC621F/V4VQAAAAAAAADgIt0JUu8OVREAAAAAAAAA4GbdCVJHS3rfWrs9VMUAAAAAAAAAgBt1J0gdLulRY0yOpNWS3pe0zFq7LgR1AQAAAAAAAIBrdDlItdbOlyRjTJykayT9VNLvJEWEpjQAAAAAAACg/zCecFeAI+lykGqMuVPSaZISJK2VdLOkZSGqCwAAAAAAAABcozu39n9dkl/Sm5KWSlphrfWGpCoAAAAAAAAAcJEudxi21k6TNFfSx5LOkbTeGPNBqAoDAAAAAAAAALfozq39EyWdIWm2pOmS8sWt/QAAAAAAAAD6ge7c2v+QpCWSHpa01lpbH5qSAAAAAAAAgP7HmHBXgCNxvLXfGBNpjPmVpBMUHCf1IUm7jDG/MsZEhbpAAAAAAAAAAAi3royR+oCkgZJGWmuntY2VOkpSiqRfh7I4AAAAAAAAAHCDrgSpX5F0jbW27vMF1tpaSddJuiBUhQEAAAAAAACAW3RljFRrrbWHWNhqjDloOQAAAAAAAIDu8zBGqqt1pUfqJmPMFQcuNMb8l6QtPV8SAAAAAAAAALhLV3qkXi/pFWPMdyStbls2XVKcpItDVRgAAAAAAAAAuIVjkGqtLZR0sjHmLEkT2hYvstYuDmllAAAAAAAAAOASXemRKkmy1r4r6d0Q1gIAAAAAAAAArtTlIBUAAAAAAABA6Bgmm3K1rkw2BQAAAAAAAAD9GkEqAAAAAAAAADggSAUAAAAAAAAABwSpAAAAAAAAAOCAyaYAAAAAAAAAF2CyKXejRyoAAAAAAAAAOCBIBQAAAAAAAAAHBKkAAAAAAAAA4IAxUgEAAAAAAAAXMAyS6mr0SAUAAAAAAAAABwSpAAAAAAAAAOCAIBUAAAAAAAAAHBCkAgAAAAAAAIADJpsCAAAAAAAAXMDQ5dHVeHsAAAAAAAAAwAFBKgAAAAAAAAA4IEgFAAAAAAAAAAcEqQAAAAAAAADggMmmAAAAAAAAABcwJtwV4EjokQoAAAAAAAAADghSAQAAAAAAAMABQSoAAAAAAAAAOGCMVAAAAAAAAMAFGCPV3eiRCgAAAAAAAAAOCFIBAAAAAAAAwAFBKgAAAAAAAAA4IEgFAAAAAAAAAAdMNgUAAAAAAAC4AJNNuRs9UgEAAAAAAADAAUEqAAAAAAAAADggSAUAAAAAAAAABwSpAAAAAAAAAOCAyaYAAAAAAAAAF/Aw2ZSr0SMVAAAAAAAAABz0So/UMUktvfFj4GL5tRHhLgFhVvFqfrhLQBjlzL8u3CUgzHbf80i4S0CYTXiS80B/ZwN0senv6v304+nvEhLDXQEAHB0+yQAAAAAAAADAAWOkAgAAAAAAAC5guIHD1eiRCgAAAAAAAAAOCFIBAAAAAAAAwAFBKgAAAAAAAAA4IEgFAAAAAAAAAAdMNgUAAAAAAAC4AJNNuRs9UgEAAAAAAADAAUEqAAAAAAAAADggSAUAAAAAAAAABwSpAAAAAAAAAOCAyaYAAAAAAAAAFzAeZptyM3qkAgAAAAAAAIADglQAAAAAAAAAcECQCgAAAAAAAAAOGCMVAAAAAAAAcAHDEKmuRo9UAAAAAAAAAHBAkAoAAAAAAAAADghSAQAAAAAAAMABQSoAAAAAAAAAOGCyKQAAAAAAAMAFmGzK3eiRCgAAAAAAAAAOCFIBAAAAAP+fvTsPj6q8Fzj+fSckIRggAcIuq4oKKLjgWlFxq0ttvb21autSW2t3u2mrt1Ztbb3dl9vaWlv3qmhttWrVioDWXRAEVJB9D5CFLXvy3j8mBgKYGZYkA3w/z8OTOed9z5nf8RxnzvnNu0iSpBRMpEqSJEmSJElSCiZSJUmSJEmSJCkFJ5uSJEmSJEmSMoCTTWU2W6RKkiRJkiRJUgomUiVJkiRJkiQpBROpkiRJkiRJkpSCY6RKkiRJkiRJGSDhGKkZzRapkiRJkiRJkpSCiVRJkiRJkiRJSsFEqiRJkiRJkiSlYCJVkiRJkiRJklJwsilJkiRJkiQpAwQnm8potkiVJEmSJEmSpBRMpEqSJEmSJElSCiZSJUmSJEmSJCkFE6mSJEmSJEmSlIKTTUmSJEmSJEkZINjkMaN5eiRJkiRJkiQpBROpkiRJkiRJkpSCiVRJkiRJkiRJSsExUiVJkiRJkqQMEEJ7R6CW2CJVkiRJkiRJklIwkSpJkiRJkiRJKZhIlSRJkiRJkqQUTKRKkiRJkiRJUgpONiVJkiRJkiRlgOBsUxnNFqmSJEmSJEmSlIKJVEmSJEmSJElKwUSqJEmSJEmSJKVgIlWSJEmSJEmSUnCyKUmSJEmSJCkDONdUZrNFqiRJkiRJkiSlYCJVkiRJkiRJklIwkSpJkiRJkiRJKThGqiRJkiRJkpQBHCM1s9kiVZIkSZIkSZJSMJEqSZIkSZIkSSmYSJUkSZIkSZKkFBwjdSfFGJl610OsmDaLrJwcjvrCp+k2eECzOnXVNbz4q9vZsGoNIQT6HT6SQy/4aLM6S159kxd/dTun/fBqug0d2JaHoB0wpqiAr40cQiIEHl9UzH1zlzYrz04Erht9AMMK8llXU8f333iXlZXVTeU983K556TDuGP2Yh6YtwyATwzpy9kDehGB+esq+PG0OdQ0xLY8LO2gEw7pw/c+fRhZicCDk+bxx3++06z8v04YzDUXjKK4rBKAe56Zw/hJ8zloYAE3XXYk+XnZNDREfv/oLJ54ZXF7HIJ20vH9Crn26KEkEoGHZ6/k9reWNCs/ondXvnvUEA7ols83J77DMwvXNJV968jBjN23GyEEXlpWxo9emdfW4auV/eGnn+fD40azumQdR5x6dXuHo10oxsiK8Q+wftYMEjk59L/4MvIGbH0fV7loEUvuvoNYW0Pn4SPp84lPEhoHQFszcQKlkydBItB5xCH0Oe/jTdvVlJbw3k3fp+dZ51B06ultdVjaQUf1LOCqQ5L3h/9cVMy9c7a+P/ze4cn7w7U1dVz/+rusrEjeHw7t0omrR+/HPh2yaIjw2UnTvA/MQF88aDBjehRS3dDAT2e8x9x1G7eqs3+Xffj2yP3JSSR4bU0Zv39nAQCdsztw3aHD6J2Xy8rKan447V021NV/4H57dszlhtEHkgiQFRI8ungFjy9ZCcDY3j24cGh/EgReXV3K7XMWtd1/BLVoTFEBXx0xhESAJxYXc9/cZc3KsxOB60YdwAEF+7Cupo4bpsze4jkxh7tPPIw7Zy/mgfnL2zp8SWmwRepOWjFtFhtWruasX97AkZ+7kDf+/MA26x149imc9fPrOf2W77J69nyWT5vVVFZbWcWcpybSfb9BbRS1dkYC+MYhQ/nWK7P49HNTOaVfEYPy85rVOWtAL9bX1nHBhCmMn7eMKw8e1Kz8K8MH8+qqsqblHh1z+K/Bffns89O5ZNKbJAKM61fUBkejnZUIgRsuPZzP/GQSp1/9JOccM5D9+nXZqt4TryzmnGuf4pxrn2L8pPkAVFbX8+1bX+bD1zzJZf87if/51GF07pTd1oegnZQI8L1j9+OKZ2Zyzt/e4KwhRQwt6NSszvINVXz3+Tk8MW9Vs/WjenZhdK8unPv3KXzkkTcYWdSZI3t3bcvw1QbueWgy5158S3uHoVawftZMqlet4oAbb6bfhZ9m2f33bbPesvvvpf9Fn+aAG2+metUqNsyaCcCG2e+ybvp09rvueg64/iaKTjmt2XYrHh5P/vARrX4c2nkJ4JuHDuWbL83iomenckr/IgZ1bn5/ePbA5P3h+f+ewoNzl/HF4YMAyApw/RHD+Omb8/jUhDf58n9mUGcSNeOM6VFIv055XPrCVH41cy5fPXjoNut99eCh/HLmXC59YSr9OuVxZI8CAM4f3I83S8q59IWpvFlSzieH9G9xv6XVNXztlbe48qXpfOWV6Zw/pB/dc3PonN2BK4YN4urXZvK5F9+kMDeH0d28d8gECeDrI4fw7VdncfHENxnXt4iBWz4n7pv8HLjwuamMn7+cKw8a1Kz8ywc3f07U3imEvevf7sZE6k5aNuUtBn3oKEII9Nh/MLUVlVSWrW1Wp0NuDr2GHwBAVocOdBu8L5Ul5U3lM8Y/zkHnnEYi2wTK7uCgws4s21jFiopq6mJkwrLVHN+7e7M6H+rdnaeWJBMmk1as4fDGG6hkWTdWVFSxYH1Fs22yEoHcrARZATpmZbGmqqb1D0Y77dCh3VhUvIElqzdSW9/A468s5pTD+6e17cKV61lYvAGAVeWVlKyronvn3NYMV63gkKLOLF5XydL1VdQ2RJ6cv5qTBzT/TFi+oZo5ZRtpiFs+GEdysxJkJxLkJBJ0CIGSSv/f39O8+Nq7lJZvaO8w1ArWT59G4dFHE0Kg05Ch1FdUULu2vFmd2rXlNFRV0WnIUEIIFB59NOumTwOg9PlJ9Dz9jKZ7wA5dNv0Qt3bam+R070HHPn3b7oC0ww7q1pmlG6tY/v794dLVfKjPFveHfbrz5OLG+8Plazi8KHl/OKZnIfPWbmxq3biupo6Gtg1faTimVzeeXZ48f++s3UB+dge65TZ/fuuWm02nDlm8szb5mf/s8lUc2yt5HRzbqzv/btz+35ut/6D91sVIbeN9Q3YiQYJktqFPXkeWVVSytrYOgDdLyrd6FlH72Oo5cflqju/drVmd43t346mlyfM9ecUaDivq2qxsRUU1C7d4TpSUWUyk7qTK0rV06r4pSZbXrYDK0vIPrF+zsYJlU2fQa8QwAEoXLKaitIy+h9naYHdR1DGHVZt1v1hdVU2PvJxmdXpsVqc+wsa6OrrmdCAvK8GF+/XnjtnNu2+vqarhgbnLePjUI/nHaUexoa6O11d/8HWkzNGrWydWlGy62VlZWkGvwryt6p1x5L488eMP839fO44+3TptVX7IkG5kd0iwaJXJlt1Nz065rNy46TOhuKKaXvvktLDFJtNWrefVFeU8f8HRPH/h0fxnWRnz11a2VqiSdrHa8jKyCzc9JGcXFlJbvkUitbycDgWFm+oUFFJbnmxtVL2qmI1z32Pu//6I+b/4KRULk12A66uqWP3MU/Q865w2OArtClveH66qrKaoY/PvgqK8HFZVbHZ/WJu8P9w3P48I/OLY4fzlpFFcuH+/tgxdaeqR2/wcr6mqpkdu7hZ1cps1hlhdVUOP3OR1UJiTTWl1LQCl1bUU5mSn3G9Rxxz+eNwo/nriETy4YCkl1TUsr6ik/z559MrLJRHg2J7dKOroD/GZIPkM2Pz8b3lutnpOrN3sOXFoP+6c4zBfUqZLK5EaQugVQvhzCOFfjcsHhxAub93Q9jwN9fW8/Ns7OOD0E8nv1YPY0MCb9zzCqE+d196hqY1cNmwA4+cvp7K+eTuD/Owsju/djfOffZ2PPvMaeVlZnNbfrv17iglTlzH2qsc467v/4sUZK/nplUc3Ky8q6MjPv3AM19z2Kls1WNQebUDnjgwt6MRJD7zCife/wtF9Czi819ZDQ0jaM8X6BuorNjL06u/S+7yPs/j2PxJjZNUT/6THuFPI6tixvUNUG8gKgUO6d+HGN2bzheffYmzf7hxeZFftPV06t3yrq2r4/IvTuPT5qZzatycFOdlsqKvnN7Pmcd2hw/jlUSMprqzeRo8X7W4uGzaAh7bxnCgp86Q72dSdwB3AdY3Lc4AHgT9/0AYhhCuAKwDOuu4qDj/vrB2PMsO898xk5j33IgDdhgykYrNu+pWl5eR1K9jmdq//6a/k9y5i2JknA1BbVc3aJct57qZfAVC1dh3P/+yPnPCtzzvhVAZbXVVDz7xNvywWdcxlzRZdcdc01lldVUNWgH06dGBtTR0HF3bmxL49+MLBg8jP7kCMkZr6Bkqra1hRUUV5TbKLzuQVJYwo7MIzS1e36bFp+xWXVtCn+6YWpr27dWqaVOp95Rs2XR8PTpzPNReMalrOz+vA7d8ay88feotpc0taP2Dtcqsqqum9z6bPhF6dcinemF73/FMG9WD6qvVU1CVvml9YUsqonl2YUryuVWKVtPNKJk2k9MXnAcgbOJjastKmstqyMrILmt8HZhcUUFe+aby72vIyshtbqGYXFtJl1GHJoQEGDSaEBPUbNlCxYD5rp05h5SN/o76yghACITubHiee3AZHqB2x5f3h+/eBzepU1tCz02b3h9nJ+8NVldVML1nL2sb7wJdXljGsIJ8pq5sPF6a295EBvTmzfy8AZq/dQM+8XGaVrwegR8dc1lRXN6u/prqaHpu1RC7qmMOa6uR1UFZTS7fcZKvUbrnZlNfUNm5Tk3K/JdU1LNxQwcjCLrxQXMIrq8t4ZXXyc+XM/r2oN5GaEZLPgM3P/+qq6m3U2fpz4KCCfMb26c6Vmz8nNjTwyMKVbX0YklJIN5HaI8Y4PoTwXYAYY10Iob6lDWKMtwG3AXx/6rN71Cf7/qeNZf/TxgKwfOpM3ntmMgOOPZySuQvJ7pRHXuHWvyC/9eA/qa2sYswVFzWty+mUx3l/+knT8oSbfsXoiz5mEjXDvVu+nv775NGnUy6rK2sY16+IG6fOblbnPytLOWPfnswqW8+JfXowdU0y2f7lF2c01bls2AAq6+p5ZOEKDi7IZ3hhZ3KzElTXN3B4UVdmO57ebuGt+aUM6t2Z/kX7UFxaydlHD+Drv3upWZ2igo6sLq8C4JTD+zF3eTJJlp2V4NarPsTf/7OQp15bstW+tXuYsXo9A7vk0S+/I6sqqjlzSBHfnvRuWtuu2FDNfw/rzW0BAoEj+nTl7pnLUm8oqd10P/Ekup94EgDrZrxFyaSJdD1iDJUL5pOVl0d21y0SqV0LSHTsSMX8eeQNHkLZK6/Q/aRkQrTLoaPYOGc2+cMOpLp4JbG+jqz8fIZ+65qm7Ysff4xEbq5J1Az3btl6+udvdn/Yv4gbX9/i/nBFKWcO6Mms0vWc2LcHUxqHcXptVRkXHdCf3KwEdQ0NjOrRlQfn+l2QCR5bvJLHFicTWWOKCjl3QB8mrljDQV3z2Vhb19RV/32l1bVU1NVzUNd83lm7gVP69uTRRSsAeHlVKaf27cmDC5Zxat+evFRc0rR+W/vtkZvDuto6ahoayO+QxYjCLvxtYXIW94KcZCI2v0MWHxnQmx9Ma36tqX00PSc2JkrH9S3ipi2eE18sLuWM/snnxLF9ejB1TfIHk6+8NLOpzmUH7Nv4nGgSdW+V2A0nYNqbpJtI3RhC6E5jD4QQwtGAP5ECfUYPZ/m0WTx+1Q10yM3hqM9/qqnsqe/8iDNuuZaKkjLe/sdTdOnbi6evTc7au/9pYxl68nHtFbZ2Qn2EX86Yx8+PHkEiwBOLi1m4voLLhw3g3fINvFhcyhOLV/I/hw3j/nGHs66mjhumtJxUebt8A5NWlPDnE0ZRHyPvrd3IY4v84twd1DdEbrzzDe685kQSicDDk+fz3rJ1XPVfI5mxoJQJU5dxyenDGHdYP+rrG1i7sYar//AKAGcePYAjD+xJQedc/uuEwQBc/cdXeGeR4+PuTuoj/PDludx+xggSIfDInJXMLa/gK4cNZOaa9UxcXMqIHvn89pThdMnpwEkDuvOVwwZyziNTeHrhao7qW8Cj5x1BjJH/LCtj0pLS1G+q3cpdv/0KHzrmIHoUdmbuq//HD37xMHc9OKm9w9Iu0HnESNbPnMGc668j5OTQ/+JLm8reu/lG9r/u+wD0veAilt51B7G2lvzhI+g8PDk2fuGxx7PsnjuZc9P3CR060P/iywi74/S1St4fTp/HL44bQRbw+KJiFqyv4LMHDeDdsg38Z2Upjy9ayfeOGMaDpybvD7//evL+cH1tPQ/MXcafTzyUSLJF6svFztqdaV5bXcZRPQq564TDqK5v4Gcz5jaV/eHYQ7nypekA/Pbt+Xxr5H7kZiV4fXU5r61JnssH5i/le6OG8eH+vSiurOaH02e3uN8B+Xl8/sDBxJic1fqhBctYuCE5Lv8XDxrMkM77AHDv3CUsq6hqs/8O+mD1EX41cz4/O3o4iQBPLlnFwg2VfGbYAGY3PScWc93oA/jryYexvqaOG6aaBJfSEUI4A/g1kAXcHmO8ZYvyXOBu4HCgBDg/xriwsey7wOVAPfDVGOPTOxVLTKMbQAjhMOC3wAhgJlAEfDzG+FY6b7KntUjV9ntuqWN87e2WP2iLy71ZzslOnLG3W3jjre0dgtrZWXd+ob1DUDtbsdZ5bvd2nfL8oWBvt0UjXu2lnj/nOD8MWjDuXy/uVTm0CR9u+XoIIWSRHGL0VGAp8DpwQYzx7c3qfBE4JMZ4ZQjhk8DHYoznhxAOBu4HxgB9gWeBA2KMLfayb0ladzMxxqnAWOBY4PPA8HSTqJIkSZIkSZK0A8YAc2OM82OMNcADwLlb1DkXuKvx9cPAuJDs5nMu8ECMsTrGuACY27i/HZZW1/4QwpbTyh8QQlgLzIgxrtqZACRJkiRJkiQ5Ruo29AM27+K6FDjqg+o0zuu0FujeuP6VLbbdqe6S6Y6RejlwDDCxcflEYAowOIRwU4zxnp0JQpIkSZIkSdLeJYRwBXDFZqtua5zAPiOlm0jtABwUYywGCCH0IjmI61HA84CJVEmSJEmSJElpa0yatpQ4XQbsu9ly/8Z126qzNITQAehKctKpdLbdLumO+L7v+0nURqsa15UCDhctSZIkSZIkaVd7HV9zGZsAACAASURBVNg/hDA4hJADfBJ4bIs6jwGXNL7+OPBcjDE2rv9kCCE3hDAY2B94bWeCSbdF6qQQwuPAQ5sFNTmEsA9QvjMBSJIkSZIkSdKWGsc8/TLwNJAF/CXGOCuEcBPwRozxMeDPwD0hhLlAKclkK431xgNvA3XAl2KM9TsTT7qJ1C8B5wHHNy7fFWN8uPH1STsTgCRJkiRJkiRIhNjeIWScGOOTwJNbrLt+s9dVwH9/wLY3AzfvqljSSqQ2Nof9W+M/QggfCiH8Lsb4pV0ViCRJkiRJkiRlqnRbpBJCGA1cAHwCWAA80lpBSZIkSZIkSVImaTGRGkI4gGTy9AJgDfAgEGKMdueXJEmSJEmStNdI1SL1XeAF4OwY41yAEMLXWz0qSZIkSZIkScogqRKp55Gc6WpiCOEp4AEgtHpUkiRJkiRJ0l4mYdYtoyVaKowx/iPG+EngQGAicBXQM4RwawjhtLYIUJIkSZIkSZLaW4uJ1PfFGDfGGP8aYzwH6A+8CVzzfnkIobCV4pMkSZIkSZKkdpdWInVzMcayGONtMcZxm62esAtjkiRJkiRJkqSMkmqM1HQ5goMkSZIkSZK0E7a7xaPa1K46P3EX7UeSJEmSJEmSMo6JbkmSJEmSJElKYVclUu3aL0mSJEmSJGmPlXYiNYRwfAjhssbXRSGEwZsVj/uAzSRJkiRJkiRpt5fWZFMhhO8DRwDDgDuAbOBe4DiAGGNpawUoSZIkSZIk7Q0SwWmIMlm6LVI/BnwE2AgQY1wOdG6toCRJkiRJkiQpk6SbSK2JMUYgAoQQ9mm9kCRJkiRJkiQps6SbSB0fQvgjUBBC+BzwLPCn1gtLkiRJkiRJkjJHWmOkxhh/FkI4FVhHcpzU62OM/27VyCRJkiRJkiQpQ6SVSAVoTJyaPJUkSZIkSZJaQSK0dwRqSYuJ1BDCepLjoobGv01FQIwxdmnF2CRJkiRJkiQpI7SYSI0xdm6rQCRJkiRJkiQpU6U12VQI4egQQufNljuHEI5qvbAkSZIkSZIkKXOkO0bqrcBhmy1v3MY6SZIkSZIkSTsorRaPajfpnp8QY2waIzXG2MB2TFQlSZIkSZIkSbuzdBOp80MIXw0hZDf++xowvzUDkyRJkiRJkqRMkW4i9UrgWGAZsBQ4CriitYKSJEmSJEmSpEySVvf8GOMq4JOtHIskSZIkSZIkZaQWE6khhKtjjD8JIfwWiFuWxxi/2mqRSZIkSZIkSXuRRGjvCNSSVC1S32n8+0ZrByJJkiRJkiRJmarFRGqM8Z+Nf+9qm3AkSZIkSZIkKfOkNUZqCOEA4FvAoM23iTGe3DphSZIkSZIkSVLmSCuRCjwE/AG4HahvvXAkSZIkSZIkKfOkm0itizHe2qqRSJIkSZIkSXuxELaa610ZpMVEagihW+PLf4YQvgj8Hah+vzzGWNqKsUmSJEmSJElSRkjVInUKEIHQuPztzcoiMKQ1gpIkSZIkSZKkTNJiIjXGOLitApEkSZIkSZKkTJXWGKkhhE7AN4ABMcYrQgj7A8NijI+3anSSJEmSJEnSXiIRUtdR+0mkWe8OoAY4tnF5GfDDVolIkiRJkiRJkjJMuonUoTHGnwC1ADHGCjaNmypJkiRJkiRJe7R0E6k1IYQ8khNMEUIYClS3WlSSJEmSJEmSlEFaHCM1hPA74H7gBuApYN8Qwn3AccClrR2cJEmSJEmSJGWCVJNNzQF+CvQB/g08C0wFvhZjXNPKsUmSJEmSJEl7jXS7jqt9tHh+Yoy/jjEeA4wF5gLnAT8HvhhCOKAN4pMkSZIkSZKkdpdWojvGuCjG+L8xxtHABcDHgHdaNTJJkiRJkiRJyhBpJVJDCB1CCOc0jo/6L2A2ydapkiRJkiRJkrTHSzXZ1KkkW6CeCbwGPABcEWPc2AaxSZIkSZIkSVJGSDXZ1HeBvwLfjDGWtUE8kiRJkiRJ0l4pEWJ7h6AWtJhIjTGe3FaBSJIkSZIkSVKmSmuMVEmSJEmSJEnam5lIlSRJkiRJkqQUUo2RKkmSJEmSJKkNJEJ7R6CW2CJVkiRJkiRJklIwkSpJkiRJkiRJKZhIlSRJkiRJkqQUTKRKkiRJkiRJUgptMtnUfbcUt8XbKIM1FOS2dwhqZ3mn9GvvENSODhzgiOl7u+F3fqG9Q1A7e+LSW9s7BLWzysU3tncIamcXT17R3iGonY0bWtXeIUgZzxaPmc3zI0mSJEmSJEkpmEiVJEmSJEmSpBRMpEqSJEmSJElSCiZSJUmSJEmSJCmFNplsSpIkSZIkSVLLEs7Tm9FskSpJkiRJkiRJKZhIlSRJkiRJkqQUTKRKkiRJkiRJUgqOkSpJkiRJkiRlgESI7R2CWmCLVEmSJEmSJElKwUSqJEmSJEmSJKVgIlWSJEmSJEmSUjCRKkmSJEmSJEkpONmUJEmSJEmSlAESob0jUEtskSpJkiRJkiRJKZhIlSRJkiRJkqQUTKRKkiRJkiRJUgomUiVJkiRJkiQpBSebkiRJkiRJkjKALR4zm+dHkiRJkiRJklIwkSpJkiRJkiRJKZhIlSRJkiRJkqQUHCNVkiRJkiRJygCJENs7BLXAFqmSJEmSJEmSlIKJVEmSJEmSJElKwUSqJEmSJEmSJKVgIlWSJEmSJEmSUnCyKUmSJEmSJCkDJEJ7R6CW2CJVkiRJkiRJklIwkSpJkiRJkiRJKZhIlSRJkiRJkqQUTKRKkiRJkiRJUgpONiVJkiRJkiRlACebymy2SJUkSZIkSZKkFEykSpIkSZIkSVIKJlIlSZIkSZIkKQXHSJUkSZIkSZIygC0eM5vnR5IkSZIkSZJSMJEqSZIkSZIkSSmYSJUkSZIkSZKkFEykSpIkSZIkSVIKTjYlSZIkSZIkZYBEiO0dglpgi1RJkiRJkiRJSsFEqiRJkiRJkiSlYCJVkiRJkiRJklIwkSpJkiRJkiRJKTjZlCRJkiRJkpQBEqG9I1BLbJEqSZIkSZIkSSmYSJUkSZIkSZKkFEykSpIkSZIkSVIKjpEqSZIkSZIkZQBbPGY2z48kSZIkSZIkpWAiVZIkSZIkSZJSMJEqSZIkSZIkSSmYSJUkSZIkSZKkFJxsSpIkSZIkScoAidDeEagltkiVJEmSJEmSpBRMpEqSJEmSJElSCiZSJUmSJEmSJCkFE6mSJEmSJEmSlIKTTUmSJEmSJEkZIITY3iGoBSZSd9IJh/bhfy47gqxEYPyEufzx0beblZ83dgjf+fRoVpZWAHDvU3MY/9w8AGY/cAGzF5cDsGJNBZ//yeS2DV67xAnDe3H9+aNIJALj/7OAPzw1e5v1zjisH7+/8hjOvXkCMxaVce6Yffnc6cOayg/s15Vzfvgs7yxd21ahaxc5vl8h3zlqKFkh8Lc5K7l9xpJm5Yf36sp3jhrCAYX5fHvSOzyzaA0AY3p35ZoxQ5vqDe7aiW9NfofnFpe0afzaMTFGVjx0PxtmzSBk59D/4s+QN2DgVvUqFy9k6d13EGtryB8+kj7/fQEhBIoff5SyF1+gQ+fOAPT6yMfoPOIQKhbOZ/lf73n/Teh51kfoMuqwtjw0pSnGyIrxD7B+1gwSOTn0v/iybV8DixaxpPEa6Dx8JH0+8UlCSE7HumbiBEonT4JEoPOIQ+hz3sebtqspLeG9m75Pz7POoejU09vqsNQK/vDTz/PhcaNZXbKOI069ur3DUSuJMXLzzbcxefIUOnbM5ZZbvsbw4fttVe/JJ1/g1lvH09BQz4knjuHb374UgOXLV3HNNb9i/fqN1Nc38K1vXcLYsUe08VFoe8UYWfTgg5TPSH4XDL30UvYZuPV3wcZFi5h3xx001NZSMHIkA88/nxACG5csYeF991FfVUVujx4MvfxyOuTl0VBXx4J772XjwoWERIKB559Pl2HDthGBMkmMkX/f9jfmvfE2HXJzOOeqi+i9375b1Xvg+t+zoXQdDQ0N7HvwUE7/wn+TyErw/H1PMu3pl+nUNR+AEy8+m/2OHN7WhyGpBSZSd0IiBG64/Egu+eFzrCyp4JEfn8GEN5Yyd9m6ZvWeeGkRN/7lja22r6qp5yNX/6utwlUrSAS48cLRXPzLF1hZVsE/rh3Hs9OXM3fF+mb19sntwKUn78eb8zclyB59bQmPvpZMuA3r14U/fPFYk6i7oUSA647ej889PYPiimoePGc0ExeXMG9tRVOdFRuruO6FOVw6on+zbV9buZb/emwqAF1zOvCvjx/JS8vK2jR+7bgNs2ZQs2oV+9/wIyoXzmf5A/cy9Orrtqq3/P576XfRxeQNGsKi3/2aDW/PpPPwkQD0OPlUemyRIOvYtx9Dr/kfQlYWtWvLmXvzjXQeeSghK6tNjkvpWz9rJtWrVnHAjTdTuWA+y+6/j/2uuXaresvuv5f+F32avMFDWPh/v2HDrJl0HjGSDbPfZd306ex33fUksrOpW9f8/mHFw+PJHz6irQ5Hreiehybzh7ue5vZffrG9Q1Erev75KSxcuJxnnvkj06fP5oYbbuWhh37erE5Z2Tp+8pO/8Mgjv6Jbt65cc80vefnl6RxzzKHceut4Pvzh47nwwjOZO3cxV1xxI8899+d2Ohqla+3MmVQVF3PoD3/IhgULWHDffYy4duvvggX33cfgiy8mf/BgZv/mN6ydOZOCkSNZcPfdDPj4x+kybBir/vMfVjzzDPueey6rXngBgENuuIHadet49ze/YcS11xISjs6Xyea98Taly1dz5W3fY/nshTz1+/Fc+otvblXvY9+5jNxOecQYeeTHf+Gd/7zJ8LGHAzDmoydy9Hnj2jp0SWnyU3gnHLpfdxatXM+SVRuorW/giZcWccqRW//apD3XoYO7sWjVBpas2UhtfeTx15dw6qF9t6r3jXOH88enZ1Nd27DN/Zxz5AAef33JNsuU2Ub26MyS9ZUs3VBFbUPkyfmrOWlA92Z1lm+oZk7ZRmL84C4apw3qwQtLy6iq3/Y1osyz7q1pFBx1DCEEOg0eSn1FBbVry5vVqV1bTn1VFZ0GDyWEQMFRx7Bu+pst7jeRk9uUNI21tRBa7RC0k9ZPn0bh0Ucnr4EhH3wNNFRV0WlI8hooPPpo1k2fBkDp85PoefoZJLKzAejQpUvTdmunvUlO9x507LP1d4p2Py++9i6l5RvaOwy1sgkTXuGjHz2ZEAKjRh3IunUbWbWqtFmdJUtWMnBgX7p16wrAMcccytNPvwhACLBhQ/KH2PXrK+jZs1vbHoB2SNm0afQ4Jnk/0HnIEOorK6kpb/5dUFNeTn1lJZ2HDCGEQI9jjqFsWvK7oKq4mM4HHABA14MPpnRq8kf2yhUrmlqgZnfpQodOndi4aFEbHpl2xJxXZzDy5DGEEOh34GCqNlayoXTrxjK5nfIAaKhvoL62rqmniqTMl7JFagjhGy2Vxxh/sevC2b306pbHipJNrc5WllRw6P7dt6p3+lEDOPKgnixcsZ6b75rStE1udhZ///EZ1NdH/vDoLJ59fWmbxa5do3dBHitKK5uWV5RXMmpw85ve4QMK6NMtj4kzVvK507bdHeesI/vz+d+91KqxqnX06pTLio3VTcvFFdUcUtR5u/fz4SE9uWumnwG7k7rycrILN/3/nl1YmFzXtaB5nYLCreq8r2Tyc5S9+hJ5AwfR578+QVanfQCoWDCfZffeSW1pCf0vudzWqBmqtrxsq2ugdotroLa8nA6bXwMFhdSWJ1ueV68qZuPc91j52D9IZGfT+7yP02nQYOqrqlj9zFMM/urXWfPsM213QJJ2SnFxCb1792ha7t27O8XFJc0SogMH9mXBgmUsXVpM7949mDDhFWpr6wD48pcv5PLLr+feex+nsrKKO+74YZsfg7ZfTXk5uYWbPudzCgupKS8np6CgWZ2cbdQByOvbl7Jp0+g2ejSlU6ZQU5pMvu/Tvz/l06fTY8wYqsvK2LhoUbJs8OA2OjLtiA0la+nSY9O579y9gPUla8lv/PFkc/d/7/esmLOIIUcczIHHjWpaP+XxF5jx3Ov02W9fxn32Y+Tld2qT2JU5EubVM1o6Xfu3PyOgJs9NWcrjLy6kpq6BT56yHz/50jF8+qYJAIz94j8oLqtk35753HP9OOYsLmdxsa0V9iQhwHX/fSjfvvP1D6xz6OBuVNXUM2f5ug+soz1bj7wc9i/sxIt269+rdD/hRHqeeQ4Aq/75D1b8bTz9P30ZAJ0GD2H/791E1YrlLLv7L+QPH9nUalF7jljfQH3FRoZe/V0qFy1k8e1/ZNgPfsyqJ/5Jj3GnkNWxY3uHKGkX69o1nxtu+CJf//pPSCQCo0cfxOLFKwB44onn+djHxvGZz3yMN998l6uv/gWPP/5/JOzKvUcbcsklLHzgAZY98QSFhx5KokPyEb3ouOOoXLGCmTffTE737uQPHQpeC3uUC37wRepqann0Z3ez6K05DB59IIedeTzHf/IMQoDJ9z7JhNv/ztlXXdTeoUraTMpEaozxxh3ZcQjhCuAKgKLDP0OXISfvyG4yWnFpJX26b/p1qHf3ThRv1joRoHxDTdPr8RPmcc2nRm/avixZd8mqDbz6djEHDyo0kbqbWVleSZ9ueU3LfQryms4rQH7HDhzQrwv3f3MsAEVdO3Lbl47lit+9xIxFyaTZOUfuyz9fs1v/7qq4opo+++Q2LffqlEvxxpoWttjaGYN7MGFRCXUtdP1XZiiZ/BxlLybHLMsbOIjask1dNmvLyuiwWesTgA4FBU2tD7es06HLppYJhcefwKLf/2ar9+vYpy+J3I5UL19G3sBBu/JQtINKJk2k9MXnAcgbOHirayB7i2sgu6CAus2vgfKyplbK2YWFdBl1WHJogEGDCSFB/YYNVCyYz9qpU1j5yN+or6wghEDIzqbHiXvevZS0u7vvvicYP/5pAEaO3J+VK9c0la1cWUKvXlv3Vjv55DGcfPIYAB588KmmROnDDz/D7bcnH71Gjz6Q6uoaysrW0b17wVb7UPtaOXEiqxvHMN1n0CCqy8qaWh/VlJU1a40KkFNQQE3Zpu+Czevk9enDQV//OgCVxcWUz5gBQMjKYuD55zdtM+uWW+jYq1drHZJ2whuPP8+0p18GoO/+A1i3ZlPvo/Ul5XTuvnVr1Pd1yMnmgKNGMueVGQwefSD5hZuG+Rl1+jGMv/G21gtc0g5Je7KpEMIBwK1ArxjjiBDCIcBHYozb7HMSY7wNuA1gv0/ct0dmB96aV8LAPp3pX7QPxaWVnHXsQL7xmxeb1Skq6Mjq8ioAxh3Rj3lLk60Ou+yTQ1V1HTV1DRR2zuXwYUX86dG32/wYtHPeWljGoJ759O/eieLySs4+cl+uuv21pvL1lXUc8Y1/Ni3/9Ztj+fHDbzUlUUOAMw/vz/k/ndTWoWsXmblmPQO65NEvvyOrKqo5c0gR35787nbt48zBPfnVlAWtFKF2pe5jT6b72GQya/2MtyiZ/BxdjxhD5cL5ZOXlNevSDZDdtYCsjh2pWDCPvEFDKH/1Zbo3JsNq127qAr5u2lQ69u0HQM2a1WQXdiNkZVFTUkJ18Qqyu2/9IK720f3Ek+h+4kkArJvxFiWTJiavgQUffA0kOnakYv488gYPoeyVV+h+UvIa6HLoKDbOmU3+sAOpLl5JrK8jKz+fod+6pmn74scfI5GbaxJVylAXXXQWF110FgCTJr3Ovfc+zllnncD06bPp3LnTNsc5LSkpp3v3Atau3cBf//okv/pV8v/5Pn2KePnl6Zx33inMm7eE6uraprFUlVl6n3QSvU9KfheUvfUWxRMn0v3II9mwYAFZeXnbTKRm5eWxfv588gcPZs3LL9P75Mb7gXXryO7ShdjQwPInnqDnCScAUF+dHDoqKzeXtW+/TcjKolNfx83OREecfQJHnJ08b3Nfn8Ubjz/PwSccxvLZC8nt1HGrbv01ldXUVFaR360rDfX1zH1jFvsePBSADaWbhgGY8/JbFA3s07YHIymltBOpwJ+AbwN/BIgxvhVC+Cuw1w7eU98QufEvb3DHdSeTlQg8NHEe7y1dy9c+cQgz55UwYcoyLvnwgYw7oh919ZG1G6q5+vfJX6qG9uvCD684ioaGSCIR+OM/3mbuMrt2727qGyI33D+Nu676EIlE4KEXF/LeinVc9ZGDmbGojAnTV7S4/Zj9i1hRVsGSNRvbKGLtavURbn5lLredNoJECPz9vZXMK6/gy6MHMmvNeiYuKWVEj3x+ffJwuuR04MR9u/Ol0QM59x9TAOibn0vvfXJ5feXWg9Ars+WPGMn6WTOY8/1rSeTkNHXLB5j7oxvZ79rvA9D3k59i6d1/oaG2ls7DR5A/fCQAK//+MFVLk63Rc7r3oO+FnwZg47y5rHnmX8lxUUOg7/mfokO+o+xkos4jRrJ+5gzmXH8dISeH/hdf2lT23s03sv91jdfABRex9K47iLW15A8fQefhIwAoPPZ4lt1zJ3Nu+j6hQwf6X3yZk03soe767Vf40DEH0aOwM3Nf/T9+8IuHuevBSe0dlnaxsWOPYPLkNzj11CvIy8vlRz/6WlPZued+lUcfTfY8uPnmP/Huu8kfUL/0pU8yeHDyh7TvfOdy/ud//o8773yUEAK33PI1PxN2AwUjR1I+cybTr7uORE4OQy69tKlsxk03MfL66wEYdOGFzL/zThpqaigYMYKuI5LfBSWvv07xxIkAFB52GEXHHQdA3fr1vPvrX0MI5BQUMPQzn2nbA9MOGXrEwcx9Yxa3fu4msnNzmnXLv/0r/8tnf3sNNVXVPPSDP1FXW0dsiAw8ZH8OOzN53p+741GK5y+DECjo2Y0Pf/n8D3orSe0ktDSLdLOKIbweYzwyhPBmjHF047ppMcZRqbbdU1ukKn0NBbmpK2mPlndMz/YOQe3owAE+CO7tQvBWYG/3xKW3tncIameVi3doxDDtQS6e3HIjA+35xvWtau8QlAEu2f90Hw5acN0bE/aqG+ebjxi3W10P2zNa9ZoQwlAgAoQQPg74TShJkiRJkiRpj7c9Xfu/RHLM0wNDCMuABcCnWiUqSZIkSZIkScogaSdSY4zzgVNCCPsAiRjj+tYLS5IkSZIkSZIyR9pd+0MIXwshdAEqgF+GEKaGEE5rvdAkSZIkSZIkKTNsT9f+z8QYfx1COB3oDnwauAd4plUikyRJkiRJkvYiCSdpzWjbM9nU+7NonQncHWOctdk6SZIkSZIkSdpjbU8idUoI4RmSidSnQwidgYbWCUuSJEmSJEmSMsf2dO2/HBgFzI8xVoQQugOXtU5YkiRJkiRJkpQ5tieRGoGDgbOBm4B9gI6tEZQkSZIkSZK0t0k4iGZG256u/b8HjgEuaFxeD/xul0ckSZIkSZIkSRlme1qkHhVjPCyE8CZAjLEshJDTSnFJkiRJkiRJUsbYnhaptSGELJJd/AkhFOFkU5IkSZIkSZL2AtuTSP0N8HegZwjhZuA/wI9aJSpJkiRJkiRJyiBpd+2PMd4XQpgCjAMC8NEY4zutFpkkSZIkSZK0F3GyqcyWMpEaQugSY1wXQugGrALu36ysW4yxtDUDlCRJkiRJkqT2lk6L1L8CZwNTSI6PGrb4O6TVopMkSZIkSZKkDJAykRpjPLvx7+DWD0eSJEmSJEmSWtbYe/5BYBCwEPhEjLFsizqjgFuBLkA9cHOM8cHGsjuBscDaxuqXxhintfSeaY2RGkLoAHwYOLBx1dvA0zHGunS2lyRJkiRJkqRd6DvAhBjjLSGE7zQuX7NFnQrg4hjjeyGEvsCUEMLTMcbyxvJvxxgfTvcN0xkjtR/wHLACeJNkl/6zgV+EEE6KMS5P980kSZIkSZIkbVtWewewezkXOLHx9V3AJLZIpMYY52z2enkIYRVQBJSzAxJp1LkZuDXGeGKM8esxxqtijGOB3wE/3pE3lSRJkiRJkqSd0CvGuKLx9UqgV0uVQwhjgBxg3marbw4hvBVC+GUIITfVG6bTtf/oGOOlW66MMf4mhDA7je0lSZIkSZIkqZkQwhXAFZutui3GeNtm5c8Cvbex6XWbL8QYYwghtvA+fYB7gEtijA2Nq79LMgGbA9xGsjXrTS3Fm04itbKFsoo0tpckSZIkSZKkZhqTpre1UH7KB5WFEIpDCH1ijCsaE6WrPqBeF+AJ4LoY4yub7fv91qzVIYQ7gG+lijedRGrXEMJ524qD5IxXkiRJkiRJknZS4oMbVWprjwGXALc0/n10ywohhBzg78DdW04qtVkSNgAfBWamesN0EqmTgXM+oOz5NLaXJEmSJEmSpF3pFmB8COFyYBHwCYAQwhHAlTHGzzauOwHoHkK4tHG7S2OM04D7QghFJBuLTgOuTPWGKROpMcbL0ok8hHBJjPGudOpKkiRJkiRJ0o6KMZYA47ax/g3gs42v7wXu/YDtT97e90xs7wYt+Nou3JckSZIkSZIkZYxdmUgNu3BfkiRJkiRJkpQx0hkjNV2OhitJkiRJkiTtoITNFDOaLVIlSZIkSZIkKYVdmUh9cRfuS5IkSZIkSZIyRtqJ1BBCrxDCn0MI/2pcPjiEcPn75THGL7dGgJIkSZIkSZLU3ranReqdwNNA38blOcBVuzogSZIkSZIkSco02zPZVI8Y4/gQwncBYox1IYT6VopLkiRJkiRJ2qs42VRm254WqRtDCN2BCBBCOBpY2ypRSZIkSZIkSVIG2Z4Wqd8AHgOGhhBeBIqAj7dKVJIkSZIkSZKUQdJOpMYYp4YQxgLDgADMjjHWtlpkkiRJkiRJkpQh0k6khhA6Al8EjifZvf+FEMIfYoxVrRWcJEmSJEmSJGWC7enafzewHvht4/KFwD3Af+/qoCRJkiRJkqS9TZaTTWW07UmkjogxHrzZ8sQQwtu7OiBJkiRJkiRJyjSJ7ag7NYRw9PsLIYSjgDd2fUiSJEmSJEmSlFm2p0Xq4cBLIYTFjcsDgNkhhBlAjDEessujkyRJkiRJKO4yHAAAIABJREFUkqQMsD2J1DNaLQpJkiRJkiRpL5dwjNSMtj2J1K8Cf44xOi6qJEmSJEmSpL3K9oyR+g7wpxDCqyGEK0MIXVsrKEmSJEmSJEnKJGknUmOMt8cYjwMuBgYBb4UQ/hpCOKm1gpMkSZIkSZKkTLA9LVIJIWQBBzb+WwNMB74RQnigFWKTJEmSJEmSpIyQcozUEMKPYozXhhB+CZwNPAf8KMb4WmOV/w0hzG7NICVJkiRJkqQ9XSLE9g5BLUinReoZjX/fAkbFGD+/WRL1fWN2bViSJEmSJEmSlDlStkgFskIIhcCjQG4IIXfzwhhjaYxxbatEJ0mSJEmSJEkZIJ1E6oHAlMbXYYuyCAzZpRFJkiRJkiRJUoZJJ5H6doxxdKtHIkmSJEmSJEkZKp1EqiRJkiRJkqRWltiyL7gySjqTTf06nR2FEH67k7FIkiRJkiRJUkZKmUiNMd6Z5r6O27lQJEmSJEmSJCkzpdMiVZIkSZIkSZL2ao6RKkmSJEmSJGWArPYOQC3alS1SHQ5XkiRJkiRJ0h5pVyZS05qUSpIkSZIkSZJ2Nym79ocQ/gnEDyqPMX6k8e+duy4sSZIkSZIkScoc6YyR+rNWj0KSJEmSJEmSMljKRGqMcfL7r0MIecCAGOPsVo1KkiRJkiRJ2ssknIEoo6U9RmoI4RxgGvBU4/KoEMJjrRWYJEmSJEmSJGWKdLr2v+8GYAwwCSDGOC2EMDidDf/2h67bHZj2LOtq/Ellb/e3hXXtHYLa0RvFOe0dgtpZbPB7YG9XufjG9g5B7SxvwPfbOwS1s5E//1J7h6B29vy/G9o7BGWAS37Y3hFIOy7tFqlAbYxx7RbrPnASKkmSJEmSJEnaU2xPi9RZIYQLgawQwv7AV4GXWicsSZIkSZIkScoc25NI/QpwHVAN3A88DfygNYKSJEnS/7N33/FV1/fix1+fBDKYCQQQZKMCMgQH1r17q112XXu11bb683rtHtpW66ra9ra32mpv9aodWm2dbR1d1q04AVkiKFtWEsgmO/n8/jghgiDnpOYkB/J68jiPfMfn+z3vL/mcb855n8+QJElST5MV7PydyVJOpMYYa0kkUi9NXziSJEmSJEmSlHlSTqSGEJ5kF2OixhhP7NSIJEmSJEmSJCnDdKRr/7e2W84DPgE4DbckSZIkSZKkvV5HuvbPfcem2SGElzs5HkmSJEmSJKlHyg7dHYF2pyNd+wdtt5oFHAIM7PSIJEmSJEmSJCnDdKRr//YtUpuBVcC5nRuOJEmSJEmSJGWepInUEMLoGOPaGOO4rghIkiRJkiRJkjJNVgpl/rxtIYTwQBpjkSRJkiRJkqSMlErX/u2HuR2frkAkSZIkSZKknizLyaYyWiotUuO7LEuSJEmSJElSj5BKi9SDQghVJFqm5rct07YeY4wD0hadJEmSJEmSJGWApInUGGN2VwQiSZIkSZIkSZkqla79kiRJkiRJktSjpdK1X5IkSZIkSVKaOdlUZrNFqiRJkiRJkiQlYSJVkiRJkiRJkpIwkSpJkiRJkiRJSThGqiRJkiRJkpQBHCM1s9kiVZIkSZIkSZKSMJEqSZIkSZIkSUmYSJUkSZIkSZKkJEykSpIkSZIkSVISTjYlSZIkSZIkZYDsELs7BO2GLVIlSZIkSZIkKQkTqZIkSZIkSZKUhIlUSZIkSZIkSUrCRKokSZIkSZIkJeFkU5IkSZIkSVIGsMVjZvP3I0mSJEmSJElJmEiVJEmSJEmSpCRMpEqSJEmSJElSEo6RKkmSJEmSJGWArNDdEWh3bJEqSZIkSZIkSUmYSJUkSZIkSZKkJEykSpIkSZIkSVISJlIlSZIkSZIkKQknm5IkSZIkSZIygJNNZTZbpEqSJEmSJElSEiZSJUmSJEmSJCkJE6mSJEmSJEmSlISJVEmSJEmSJElKwsmmJEmSJEmSpAyQHWJ3h6DdsEWqJEmSJEmSJCVhIlWSJEmSJEmSkjCRKkmSJEmSJElJOEaqJEmSJEmSlAGyQndHoN2xRaokSZIkSZIkJWEiVZIkSZIkSZKSMJEqSZIkSZIkSUmYSJUkSZIkSZKkJJxsSpIkSZIkScoATjaV2WyRKkmSJEmSJElJmEiVJEmSJEmSpCRMpEqSJEmSJElSEiZSJUmSJEmSJCkJJ5uSJEmSJEmSMoCTTWU2W6RKkiRJkiRJUhImUiVJkiRJkiQpCbv2v0cxRn5z/Z959fnXyc3L4cLLPs34iSN3Knflhb+kfEsVObm9Afjez85n4KD+7ftffHIh111yOz/89deYMHlUl8Wv9y7GyB9u+BOLXnqdnNwcvvDd/2DMATvXgW1u/O6vKN24he//9mIAbr7yDorfKgGgtqaOPv3yueJX3+qS2NU5Yoy8fte9lC54jeycHKb9v7MZOHb0TuXeuP9B1s9+iaattbz/lp+1b3/9rvvYsvQNAFoaGmmsruaUm67rsvj1rzl8aAFfnTaeLAKPrC3mzjfX7bC/d1bgewcfwMSB/ahqaubyV5ayqa6BffJzueukg1lbUwfAa2XV/M/CFQD0CoFvTJ/AzKKBtMbILa+v4emNW7r82tRxhw8t4GvTx5MVAg+vKebON3auD5cdcgATC/pR2dhWH2obAJgwoA8Xz9yPvr2yaY1w3lPzaWyN3XEZeg9ijFx77S08/fRc8vJy+dGPvsqUKfvtVO6vf32Wm266l9bWFo4/fhYXXfQ5ADZsKOHb3/4Z1dVbaWlp5VvfOofjjju0i69C6XLzT/6TU0+aSemWKg495eLuDked6H3DCvjmzMT9/8GVxdyxbOf7/5WzDmBSYT8qG5q59MWlbKxtoFcIfPeQ/Zg8qB8xwk/nr2ReaSUA/zV1DKeNGUr/nF4c/6cXuuOy9C86bv8iLj9tMtlZgXvmruOmZ1busP+sw0bx2cPH0BojWxub+e6fX2N5aQ0Ak4b15wcfnUK/3F60Rvjozc/T0NzaHZchaTdMpL5Hr76wlE1vbeaG+77Lm6+t5bYfP8APfvXVXZb9ypVn7TJJWre1nr/d+yz7T9k58aLMt+il1ylZt5kf3HUJK5es4c7r7ufSm7+2y7Jzn1lIbn7ODtsuuPLs9uV7/vdB+vTNS2u86nylC19j66YSjv3xVVSsWMVrt/+BI6/49k7lhsyYxuiTj+eZi6/YYfvksz7Vvrz6n09SteattMes9yYL+Mb0CXz9+cWU1DVy23EzeG7TFlZX17WX+dDoYVQ3NvPpx+dy0r5F/NeUsVwxZxkA67fW8/mn5u903rMPGEV5QyP/8fhcAjAgxz/Te4Is4JsHTeBrs9vqwwkzeG7jO+rDmGFUNzVzxj8T9eHCKWO5/JVlZAe4/NCJXD3nDZZXbWVATi+aTaLukZ55Zi6rV2/g0Uf/jwULlnHllTdx330/3aFMeXkVP/7xr/njH3/GoEED+fa3r+eFFxZwxBEHcdNN93LqqUdz5pmnsXz5Ws4//yqeeOJX3XQ16my/u+9pbr79H9x2/YXdHYo6URZw8cET+NIziympbeT2k2fw7IYtrNru/v+RcYn3A5/421xOGVXEl6aP5dIXl3H6+H0AOPPRVynM7c3PjpnC5x6bTwSe3VDGvcs38MCpfpmyJ8kK8P0PT+Ezv3mZTVX1PHTBkfzz9ZL2RCnAgws3ctcriff6J08aymWnTuKcO+aQnRW4/lPT+cb9C3l9UzUF+b1pajGJ2lNlO0ZqRrNr/3s055nFHHvqIYQQOGDqGLbW1FG+uapD57jnlr/z0c+cQO+c3mmKUuk0/7nFHPFvhxJCYMKUsdTW1FGxZec6UF/bwD/vfZoPnX3KLs8TY2TOkwuYdfLB6Q5Znaxk3gL2Pep9hBAo3G88zbW11FdU7lSucL/x5BUM3O25Nr44hxHvOyxdoaqTTC7sz7qt9WyobaA5Rh5bX8rR+wzeoczRwwfzt7bW5k9t2MwhRQVJz/vBMcP4XVvL1ghUNjZ3euzqfJMH7VgfHl9XyjHDd6wPxwwfzF/XblcfhiTqw6yhhayo3Mryqq0AVDU248emPdPjj7/I6aefSAiBGTMmUVW1lZKSsh3KvPXWJsaMGcGgQYm/BUcccRD/+MdsAEKAmppaAKqraxk6dFDXXoDSavbLSymrqEleUHuUKYP6s66mng1bE/f/R98q5dh9d7z/HzdiMH9Znbj/P7FuM4cNTdz/xw3IZ05JBQDlDU3UNDYzubAfAIvLqtlS39SFV6LOMGNkAWu2bOWt8jqaWiIPL9rI+ycP3aFMTcPb7+365GSz7avTY/YrYummal7fVA1ARV0Tfq8qZSaburxHZaWVFA17+8Px4CEDKSutpLBowE5lf3nN3WRlZ3H48dP5xOdPJoTAymXr2FxSwcFHHchDdz3VhZGrs1RsrmLQ0LfrQOGQAipKKykYvGMd+POv/8b7//04cnJz3nkKAN5cuJIBg/oxbOSQtMarzldfXkHe4ML29bxBhTSUVyRNmr5T3eYt1JVuZvCBEzs7RHWyIXk5lNQ1tK+X1jVwYGH/dy3TEmFrczMD21qYDu+Tx6+Pm8HW5hZufX0NC8uq6NcrG4DzJo1hZtFANmyt57pFKyhv8INUpntnfSipa2DKO+tDfg4ltdvVh6ZEfRjVL58IXHfkFApye/PYulJ+/+b6rgxfnaS4eAv77FPUvr7PPoMpLt6yQ0J0zJgRrFq1nnXritlnnyIef/xFmpoSH6q/9KUzOffcy7nzzkeoq6vnN7+5psuvQVLHDMnPobh2u/t/bQNTBu98/y/e7v1ATdv9/82KrRw7YjCPvlXKsPxcJhX2Y1ifXJaUm3DfUw0bkMeGyvr29Y1V9cwYufMX6Z89fDTnHTWO3tmBM3/9MgDjB/clAneccyiD+ubw8MKN/N9zq7oqdEkdkDSRGkJYBOzqu5AAxBjj9E6Pai/0lSvPYtDQgdRtreenl9zOM3+byzEfOJg7fv4QF1726e4OT2m29s31lK7fwqe/dDqbN5btssxLj73KrJNsjdqTbXhpDvscdjAhy84Ce7MtDY184tFXqGpqZuLAvvzg8AP57BPzyM4KDMvPZXFZFb94bRVnTBjBF6eM45p5b3R3yEqj7BCYPngA5z01n/qWVm44eirLKmqYW7pzq3bt+QYO7MeVV17I17/+Y7KyAjNnTmbt2o0A/OUvz/Cxj53EF77wMV59dSkXX3wdjzzyC7L8myDtlR5eXcy4AX24/eQZbNzawMItVbRGmyD2BL97aS2/e2ktH5k+nC8fP4FvPrCI7KzAYWMK+chNz1PX1MLvPz+LRRuqeH6lY+VLmSaVFqkf+ldOHEI4Hzgf4HvXfZFPnvOBf+U0Genv9z/H4w+9BMCEyaPYXFzRvm9LaSWDhuzcCm3Q0MS2/L55HP3+mSxfspbDjp3CWys3ctWFvwSgoqyaH1/8ay7+8ReccCrDPfGn53j2kRcBGDtxFGUlb9eB8tIKCt5RB1a8tprVy97i22dcTWtLK1XlNfz4q//LxT//IgAtzS3Me3Yhl93yja67CL0nax57ireeTnTHHDhuDPVbytv31ZeVk1uYvBv3O218cQ5TzvaLlT1BaX0jQ/Nz29eH5OdSWt+4yzKl9Y1kB+jbq1d7V/2m1sTPZZVb2bC1nlH98llWUUNdc0v75FJPrt/Mh0YP66Ir0nvxzvowdFf1oa6RoX22qw+9E/WhpK6BBVsq2+vGC5vKmVjQz0TqHuKuu/7Cvff+A4Bp0/Zn06bN7fs2bdrCsGGDdzrmxBNnceKJswC4556/tydK77//UW677SoAZs6cRENDI+XlVQwe3PG/J5K6RmldI8P6bHf/75NLad3O9/9h+bmU1CXu//16v/1+4PoFb7c4vO2E6azdbmxV7XmKq+oZMfDt+S6GD8ijuKr+Xcs/vGgj13xkCrCITVX1vLy6jPLaRE+kJ98oZeqIASZSpQyUNJEaY1yzbTmEMAzYNnjfyzHGkt0cdwtwC8CCskf2qq/WPvDJo/nAJ48GYN7sJfz9/tkcdcpM3nxtLX365u3Urb+luYWtNXUMKOhHc3MLc2e/zrRD96dPv3x+9fer28tdeeEv+eyXP2wSdQ9w4seO5sSPJerAwheW8MQfn2PWSTNZuWQN+X3zdurWf8LpR3HC6UcBsHljGTd897b2JCrA63PfYPjooTsMEaDMNubk4xlz8vEAlMxfxJrHnmL4+w6lYsUqeuXnd7hbf82GTTTX1lKw3/g0RKvOtrSimlF98xne9oHp5H2HcNXcZTuUmb2pjFNHDeW18mqOH1HEvM2JL1wKcnq1j4M5ok8uI/vmsWFrffsxM4sGMm9zJYcMKdhhsiJlrqXl1Yzs93Z9OGnkEK56Zcf68NzGMk4bPZTXyhL1YW5poj68XFLOWQeMJDc7i+bWVmYUDeSe5Xbt31OcddYHOeusDwLw1FOvcOedj/DBDx7LggXL6N+/zy7HOd2ypYLBgwuorKzh97//Kz/7WWJywuHDh/DCCwv4+MdPZsWKt2hoaGofS1VSZlpSXs2ofvmM6JNIlL5/1BAue2nH+/8zG8r44NihLCqr5sSRRe3jouZmZxGA+pZWZg0toCXGHSap0p5nwfpKxg7uy8jCfIqr6vnwtOF85b4FO5QZO7gPq7ckxsM+8YCh7ctPv1nKfx4zjrzeWTS1RA4fN4hfzV7d1ZegDJEV9qoU2l4n5TFSQwj/DvwEeIpEt/4bQwgXxRjvT1Nse4SZR05m3vOv85VP/ZCc3N5c+L23W5NddPZP+ckd36SpqZlrv3YrLc0ttLa2Mu2wAzj5o+/rxqjVmaa9bzKLXnydS878ATm5vfn8d/6jfd9V5/4PV/zqW0nP8fIT8+3WvwcbctBUShcu5umLLic7N4fp553dvu+5y67l6KsvBWDpPX9kwwuv0NLYyBNf+y6jjjuK/T+WaPS/8aU5DD88MWmZMl9LhOsWruC6I6aSFeAva4tZVV3LuZNGs7SihtmbynhkzSYuO3gid590CFVNzVw5ZykABw0eyHmTRtMcI60R/mfBCqrbxki8aclqLjv4AL4ytRcVjU388NU3u/MylaKWCNcvWMF1R00lG3hkTaI+nDd5NEvLa3huW304dCL3nHIIVY3NXPFKoj5UN7Vw9/L1/Or4g4gkWqS+UFy+2+dTZjruuEN5+uk5nHLK+eTn5/KDH3y1fd9HP/oVHnzwBgCuvfZWli5NtEL74hc/zbhx+wLwne+cy/e+9wt++9sHCSHwox991b8Je5Hbb/wyxxwxmaLC/ix/6Rdcfd393H7PU90dlt6jlgg/eXUFNxybeD/w8KpiVlbVcv6U0bxeVsOzG8t4aNUmrpo1kQdOTdz/L30xcf8flNubG46dQmtMtFq94uW3h/L58rSxvH/0EPKys3j4g4fx0Kpibl2ytrsuUylqaY1c/sgS7jjnMLKzAvfOXcebJTV8/aT9WbS+kseWlnDO4WM4asJgmlsjlXVNfPOBhQBU1Tdz2+zVPHTBkUQSLVKffKO0ey9I0i6FmOI4LCGEBcAp21qhhhCGAI/FGA9Kduze1iJVHVfV6AeBnu6B1fndHYK60ZziXU+ypp4jOhV9jzf7Y0OTF9JeLX/0Fd0dgrrZtJ9+MXkh7dVKFlR3dwjKAKuvOdUEwW48uv6vPSqH9v59T9uj6kNHRq/PekdX/i0dPF6SJEmSJEmS9kgpd+0H/h5C+Afwh7b1M4C/dn5IkiRJkiRJkpRZUkqkhsQATTeQmGjq6LbNt8QY/5SuwCRJkiRJkqSexK7fmS2lRGqMMYYQ/hpjnAb8Mc0xSZIkSZIkSVJG6Uiie14I4bC0RSJJkiRJkiRJGaojY6QeDpwVQlgDbAUCicaq09MSmSRJkiRJkiRliI4kUv8tbVFIkiRJkiRJPVxW6O4ItDsd6dp/TYxxzfYP4Jp0BSZJkiRJkiRJmaIjidQp26+EELKBQzo3HEmSJEmSJEnKPEkTqSGE74YQqoHpIYSqEEJ123oJ8GDaI5QkSZIkSZKkbpY0kRpj/GGMsT/wkxjjgBhj/7bH4Bjjd7sgRkmSJEmSJEnqVh2ZbOrSEMJngHExxqtDCKOA4THGl9MUmyRJkiRJktRjZDvZVEbryBip/wscAZzZtl7Ttk2SJEmSJEmS9modaZF6eIzx4BDCqwAxxvIQQk6a4pIkSZIkSZKkjNGRFqlNIYRsIAKEEIYArWmJSpIkSZIkSZIySEcSqTcAfwKGhhCuBZ4DfpCWqCRJkiRJkiQpg6TctT/GeFcIYS5wEhCA02OMr6ctMkmSJEmSJKkHyQqxu0PQbiRNpIYQDgduASYAi4BzY4xL0h2YJEmSJEmSJGWKVLr2/y/wLWAwcB1wfVojkiRJkiRJkqQMk0oiNSvG+M8YY0OM8T5gSLqDkiRJkiRJkqRMksoYqQUhhI+/23qM8Y+dH5YkSZIkSZLUs2SF7o5Au5NKIvVp4MPvsh4BE6mSJEmSJEmS9mpJE6kxxs+ncqIQwjkxxtvfe0iSJEmSJEmSlFlSGSM1VV/txHNJkiRJkiRJUsbozESqozhIkiRJkiRJ2iulMkZqqmInnkuSJEmSJEnqUZxsKrPZIlWSJEmSJEmSkujMROrsTjyXJEmSJEmSJGWMlBOpIYRhIYRfhRD+1rZ+YAjh3G37Y4xfSkeAkiRJkiRJktTdOtIi9bfAP4ARbetvAF/r7IAkSZIkSZIkKdN0JJFaFGO8F2gFiDE2Ay1piUqSJEmSJEnqYbJ62GNP05GYt4YQBgMRIITwPqAyLVFJkiRJkiRJUgbp1YGy3wAeAiaEEGYDQ4BPpiUqSZIkSZIkScogKSdSY4zzQgjHAROBACyLMTalLTJJkiRJkiRJyhApJ1JDCHnAhcDRJLr3PxtCuDnGWJ+u4CRJkiRJkqSeIoTujmDPEUIYBNwDjAVWA/8eYyzfRbkWYFHb6toY40fato8D7gYGA3OBz8YYG3f3nB0ZI/UOYApwI/CLtuXfdeB4SZIkSZIkSeoM3wEejzHuDzzetr4rdTHGGW2Pj2y3/b+B62OM+wHlwLnJnrAjidSpMcZzY4xPtj3+H4lkqiRJkiRJkiR1pY8Ct7ct3w6cnuqBIYQAnAjc35HjO5JInRdCeN92T3g4MKcDx0uSJEmSJElSZxgWY9zYtrwJGPYu5fJCCHNCCC+GELYlSwcDFTHG5rb1dcC+yZ4w5TFSgUOA50MIa9vWRwPLQgiLgBhjnN6Bc0mSJEmSJEnqwUII5wPnb7fplhjjLdvtfwzYZxeHXrr9SowxhhDiuzzNmBjj+hDCeOCJtlxm5b8Sb0cSqR/4V55AkiRJkiRJUnI9ba6ptqTpLbvZf/K77QshFIcQhscYN4YQhgMl73KO9W0/V4YQngJmAg8ABSGEXm2tUkcC65PF25Gu/V8B+sYY1+zq0YHzSJIkSZIkSdJ78RBwTtvyOcCD7ywQQigMIeS2LRcBRwFLYowReBL45O6Of6eOJFJfB24NIbwUQrgghDCwA8dKkiRJkiRJUmf5EXBKCOFN4OS2dUIIh4YQbmsrMxmYE0JYQCJx+qMY45K2fd8GvhFCWE5izNRfJXvClLv2xxhvA24LIUwEPg8sDCHMBm6NMT6Z6nkkSZIkSZIk6b2IMW4BTtrF9jnAeW3LzwPT3uX4lcCsjjxnR1qkEkLIBia1PTYDC0hkbu/uyHkkSZIkSZIkaU+StEVqCOEHMcZLQgjXAx8CngB+EGN8ua3If4cQlqUzSEmSJEmSJGlvF3rabFN7mFRapH6g7edCYEaM8T+3S6Ju06FmsJIkSZIkSZK0J0lljNTsEEIhiZmrcrfNdLVNjLEsxliZlugkSZIkSZIkKQOkkkidBMxtW35nA+MIjO/UiCRJkiRJkiQpw6SSSF0SY5yZ9kgkSZIkSZKkHqxDs8Kry/n7kSRJkiRJkqQkUkmk/jyVE4UQbnyPsUiSJEmSJElSRkqaSI0x/jbFcx313kKRJEmSJEmSpMxk135JkiRJkiRJSiKVyaYkSZIkSZIkpVkIsbtD0G50ZovU0InnkiRJkiRJkqSM0ZmJ1JQmpZIkSZIkSZKkPU3Srv0hhIeBd21XHGP8SNvP33ZeWJIkSZIkSZKUOVIZI/V/0h6FJEmSJEmSJGWwpInUGOPT25ZDCPnA6BjjsrRGJUmSJEmSJPUwTkCU2VIeIzWE8GFgPvD3tvUZIYSH0hWYJEmSJEmSJGWKjkw2dSUwC6gAiDHOB8alISZJkiRJkiRJyigdSaQ2xRgr37HtXSehkiRJkiRJkqS9RSqTTW3zWgjhTCA7hLA/8BXg+ZQOLO/I02hv1NTqKB89XWu0DvRkQ/r6vVtPV9Pcke9utTc6++mN3R2Cutm0n36xu0NQN1v0zf/t7hDUzQ7+ufcBKZngR+eM1pFPNV8GpgANwB+AKuBr6QhKkiRJkiRJkjJJyk1FY4y1wKVtD0mSJEmSJEnqMVJOpIYQnmQXY6LGGE/s1IgkSZIkSZIkKcN0ZPDSb223nAd8Amju3HAkSZIkSZIkKfN0pGv/3Hdsmh1CeLmT45EkSZIkSZJ6JOeaymwd6do/aLvVLOAQYGCnRyRJkiRJkiRJGaYjXfu3b5HaDKwCzu3ccCRJkiRJkiQp8yRNpIYQRscY18YYx3VFQJIkSZIkSZKUabJSKPPnbQshhAfSGIskSZIkSZIkZaRUuvZvP87t+HQFIkmSJEmSJPVkWc42ldFSaZEa32VZkiRJkiRJknqEVFqkHhRCqCLRMjW/bZm29RhjHJC26CRJkiRJkiQpAyRNpMYYs7siEEmSJEmSJEnKVKm0SJUkSZIkSZKUZg6RmtlSGSNVkiRJkiRJkno0E6mSJEmSJEmSlISJVEmSJEmSJElKwkSqJEmSJEmSJCXhZFOSJEmSJElSBgjONpXRbJEqSZIkSZIkSUmYSJUkSZIkSZKkJEykSpIkSZIkSVISJlIlSZIkSZInISZFAAAgAElEQVQkKQknm5IkSZIkSZIygHNNZTZbpEqSJEmSJElSEiZSJUmSJEmSJCkJE6mSJEmSJEmSlIRjpEqSJEmSJEkZwDFSM5stUiVJkiRJkiQpCROpkiRJkiRJkpSEiVRJkiRJkiRJSsJEqiRJkiRJkiQl4WRTkiRJkiRJUgbIcrapjGaLVEmSJEmSJElKwkSqJEmSJEmSJCVhIlWSJEmSJEmSkjCRKkmSJEmSJElJONmUJEmSJEmSlAGcayqz2SJVkiRJkiRJkpIwkSpJkiRJkiRJSZhIlSRJkiRJkqQkHCNVkiRJkiRJygAhxO4OQbthi1RJkiRJkiRJSsJEqiRJkiRJkiQlYSJVkiRJkiRJkpIwkSpJkiRJkiRJSTjZlCRJkiRJkpQBQncHoN2yRaokSZIkSZIkJWEiVZIkSZIkSZKSMJEqSZIkSZIkSUmYSJUkSZIkSZKkJJxsSpIkSZIkScoAwdmmMpotUiVJkiRJkiQpCROpkiRJkiRJkpSEiVRJkiRJkiRJSsIxUiVJkiRJkqQMYIvHzObvR5IkSZIkSZKSMJEqSZIkSZIkSUmYSJUkSZIkSZKkJEykSpIkSZIkSVISTjYlSZIkSZIkZYAQujsC7Y6J1Pcoxsjf/++PvPnKEnrn9ub0b5zF8P1G7VTuzstuoqasitaWVkZPGc9pF36KrOws7v/hb9m8vgSA+po68vrlc8EvLu7qy9B7EGPkn7c8wIo5S+iVm8OHv3YW++yiDtx9+S8TdaC1lVEHTuDf/itRB56566/M/8cL9BnYD4Djz/4Q+x02pasvQ+9BjJGld91L6cLFZOfkMO28cxgwdvRO5d68/89seP4lmrbWcvL//XyHfZtensPyPz8CBPqPHslBF5zbRdGrM8QYKb7vD1S/toisnBxGfPYL5I8es1O5urWr2fC739Da2Ej/KdMY9qn/IIRAyV8epGL2s2T36w/A0I98jP5Tp3f1ZWg3Lpw8jllFhTS0tvKTRW+yvGrrTmX2H9CXi6btT05WFi9vLueXr68CoH/vXlx60ET2yc9lU10D18xfSk1zy7ued2heLlfOnERWgOyQxYNrN/LIW5sAOG6fIs6cMJIsAi+VlnHbG2u67j9BO4kxsuaee6hYlHjtT/jc5+g7ZufX/tY1a1jxm9/Q2tREwbRpjDnjDEIIbH3rLVbfdRct9fXkFhUx4dxz6ZWfT2tzM6vuvJOtq1cTsrIYc8YZDJg4sRuuUMm8b1gB35w5nqwQeHBlMXcsW7fD/t5ZgStnHcCkwn5UNjRz6YtL2VjbQK8Q+O4h+zF5UD9ihJ/OX8m80koA/mvqGE4bM5T+Ob04/k8vdMdlKQ1u/sl/cupJMyndUsWhp/h5b29y+NACvjY9cR94eE0xd76x833gskMOYGJBPyobm7n8laVsqm0AYMKAPlw8cz/69sqmNcJ5T82nsTXy0yOnMDgvh14BFmyp4qfzV9DaHRcnaScmUt+j5XOWULa+lC/f9j3WL1vDX35xH+f97Bs7lfvUdz9Pbp88Yozcd+2vWfLcfKYedzCf/O7n2sv849Y/kdc3vwujV2dYMWcJZRtKueCWy9iwbDV//+W9fO66b+5U7mPf+Ty5ffKJMfLHH/6a1597lSnHHQLArNOP530fP6mrQ1cn2bxwMbXFJRzz39+ncsUqltzxe953+Xd2KjdkxnRGn3wCz3778h22b91UzMpH/sHhl15E7759aaiq6qrQ1UlqXltEQ2kJ+135A+pWr2Tj3Xcy/uJLdyq38e47GX7m2eSPHc/aX/6cmiWL6T9lGgCDTjyFopP/ratDVwpmFRWyb598PvfsPCYP7MdXDpzAV15cuFO5rxw4gesXL+f1yhquPeRADisq4JXNFZwxbl9e3VLBPavWc8a4ffn0+JHc9saadz1vWUMjX31xIU0xkpedxa1Hz+SFkjIaW1s5f+JYLnx+PpVNzVw0bX9mDhrIq2WV3fC/IoDKxYupLy7moGuuoWbVKlbddRdTL7lkp3Kr7rqLcWefTb9x41h2ww1ULl5MwbRprLrjDkZ/8pMMmDiRkueeY+OjjzLqox+l5NlnAZh+5ZU0VVWx9IYbmHrJJYQsR+XKJFnAxQdP4EvPLKaktpHbT57Bsxu2sKq6rr3MR8YNo7qxmU/8bS6njCriS9PHcumLyzh9/D4AnPnoqxTm9uZnx0zhc4/NJwLPbijj3uUbeODUQ7vnwpQWv7vvaW6+/R/cdv2F3R2KOlEW8M2DJvC12YspqWvkthNm8NzGLaze7j7woTHDqG5q5ox/zuWkfYu4cMpYLn9lGdkBLj90IlfPeYPlVVsZkNOL5tYIwGUvL6W27UvXa2dN4oR9i3h8/ebuuERJ77Dbd2MhhIN39+iqIDPZ0hcXM/2kwwghMHLSWOq31lG9iw80uX3yAGhtaaWl7Ya4vRgjS55NJFe1Z3njpUVMO3EWIQT2nTSO+q111OyyDiSS5K0trbQ0NRNsr7/XKHl1ISOOeh8hBAr2G09TbR0NFTvXgYL9xpNbMHCn7euefo7RJx1H7759AcgdMCDtMatzVS+cT8HhRxBCoM+4CbTW1dJUWbFDmabKClrr6+kzbkKirhx+BNULXu2miNURRwwbxGMbEr1HXq+soV/vXgzK7b1DmUG5venTK5vXK2sAeGxDCUcOGwzAkcMG88+24/+53fZ3O29zjDTFxAep3llZZJH4ezE8P4/1tXVUNjUD8OqWCo7eZ3A6L11JlM+fT9ERidd+//Hjaamro7Fix9d+Y0UFLXV19B8/nhACRUccQfn8+QDUFxfT/4ADABh44IGUzZsHQN3Gje0tUHsPGECvPn3YusbWx5lmyqD+rKupZ8PWBppj5NG3Sjl23x1fk8eNGMxfVide50+s28xhQwsAGDcgnzklibpS3tBETWMzkwsTvZMWl1Wzpb6pC69EXWH2y0spq6jp7jDUySYP6s+6rfVsqE3cBx5fV8oxw3e8DxwzfDB/XZu4Dzy1YTOHDEncB2YNLWRF5db2Xi5Vjc3trU63JVGzQ6CXX6JJGSVZi9Sf7mZfBE7sxFj2SNWbKxjYdiMEGFA0kOrNlfQftHOy5M7v3cT6N9aw3yGTOfDoGTvsW7t4BX0L+jN436Fpj1mdq2ZLJQOK3q4D/QcXUL2lkn67qAN/uOyXbHxjDeMPPZBJR71dB+Y+8iyLnniF4fuN4qTzPkZ+vz5dErs6R0N5BXmDCtvX8woLqC+v2GXSdFdqNyXeWL10zY+JrZEJp3+IIdMd3mFP0lxZQe+CQe3rvQoKaa6ooPfAt+8NzRUV9C4o3LHMdsnW8qefoPKl58kfPZZhn/h3svv07ZrglVRRbg4ldQ3t65vrGyjKzaWsoWm7Mrlsrm9sXy+tb6QoNweAwpze7WXLGpoozOmd9LxD8nK45pADGdEnj1uXrWZLQyMNLS2M7JvPsPxcSusbOHLoID9cdbPGigpyC99+XecUFtJYUUFOQcEOZXJ2UQYgf8QIyufPZ9DMmZTNnUtjWRkAfUeOpGLBAopmzaKhvJyta9Yk9o0b10VXplQMyc+huPbt13BJbQNTBvffuUzb67wlQk1TMwNzevFmxVaOHTGYR98qZVh+LpMK+zGsTy5Lyk20SXuSIXk7/i0vqWtgSuHO94GS2rfvA1vb7gOj+uUTgeuOnEJBbm8eW1fK799c337cdUdOYXJhf14sLuNJW6NKGWO3idQY4wldFUhP8Jlr/ovmxib++OM7WLXgDSYcPKl936Kn5zH1eFuj7u3+4+oLaW5s4sH/uYM1C99g3MxJHHza0Rz96Q8QAjx95195/LY/8aGvndXdoaoLxdZWaotLOOw736S+vJxXfvhTjrz6Mnr3NaHeUww65niGnPphAEof+TPFD9zLiM9+vpujUrrEFMqU1jfyn7PnMzg3hytnTuKZTVuoaGzihtdWcOlBE4lElpRXM7ytx4v2TOPPOYfVd9/N+r/8hcKDDiKrV+Kt+ZCjjqJu40YWX3stOYMH02/CBDBpvld5eHUx4wb04faTZ7BxawMLt1TRGlO5O0jaW2SHwPTBAzjvqfnUt7Ryw9FTWVZRw9y28ZK/8fxr5GQFrjh0IocMKeCV0ookZ9Tewr6rmS3lMVJDCFOBA4H2d+wxxjt2U/584HyAc6/5Mid++rT3EGZmefnhZ5n3j8TA7yP2H03ldje0qs2V9C9691ZovXJ6M/GIaSx7cXF7IrW1pYWlzy/g/BsuSm/g6jRzHnmG+dvVgarNb9eB6i0V9B+8+zpwwOHTeOPFRYybOYl+hW93457xb0dw71W3pC9wdZq1jz3FuqefA2DAuDHUl5W376svryCvsODdDt1JbmEBBRPGkdUrmz5DiugzbCi1xSUMHD+2s8NWJyp7+gnKZyfGMcwfM5amirL2fc0V5fQq2LEO9CoooKmifMcybS1Wew14+55RcNSxvHXTDekMXSn4yOh9OG3kMACWVdYwND+X1yqqASjKy2VzQ8MO5Tc3NFCUl9O+PiQvh80NiRaq5Y1NDMpNtEodlNubisamtmMak553S0Mjq2tqmVY4gGeLt/BiaTkvlibq0Wkjh9Fi4qXLbXrySUrbxjDtO3YsDeXlbGt71FhevkNrVICcggIay99+7W9fJn/4cCZ//esA1BUXU7FoEQAhO5sxZ5zRfsxrP/oRecOGpeuS9C8qrWtkWJ/c9vWhfXIprWvcuUx+LiV1jWQH6Ne7F5WNieE5rl+wqr3cbSdMZ+12YypK2jOU1if+lm8zND+X0vqd7wND+yS2Zwfo23YfKKlrYMGWyvZ7wgubyplY0K89kQrQ2Bp5dmMZxwwfZCJVyhApfbUdQrgCuLHtcQLwY+AjuzsmxnhLjPHQGOOhe1MSFWDWh4/hgl9czAW/uJhJR0xj4eOvEGNk3dLV5PbN26lbf2NdQ/u4qa0tLbz58hKKRr3dhX/lq29QNHLYDt3DldkO/dCxnHfjtznvxm9zwBHTWfTEy8QYWb90Fbl98nbq1t9Y19A+bmprSwvL57zG4LYP6NuPp/rGCwsZMmZ4112I/mWjTz6eI6/+Hkde/T2GHTyDDbNfJMZIxfKV9MrPS7lbP8DQg2dQtvQNABqra6gtLiF/aFG6QlcnGXTciUy45AomXHIF/Q+aScVLLxBjpHbVCrLy83fo1g/Qe2ABWXl51K5akagrL71A/+mJIT62H0+1esE8ckfs26XXop09tHYTFzy/gAueX8DskjJOHpH4uz15YD+2NjXv0K0fEl32a5tbmDwwMcbhySOG8kJxIrn+QkkZp7Qdf8qIoTxfvKV9+67OW5SbQ05b68N+vbKZWjiAt7YmEiwFbcMC9OuVzUdG78Pf1hWn879Bu7DPCScw7fLLmXb55RTOmMHmFxKv/eqVK8nOz99lIjU7P5/qlSuJMbL5hRconNH22m+bXDC2trLhL39h6LHHAtDS0EBLW1K9cskSQnY2fUaM6MKrVCqWlFczql8+I/rk0isE3j9qCM9uKNuhzDMbyvjg2MTr/MSRRe3jouZmZ5GXnXidzxpaQEuMO0xSJWnPsLS8mpH98hnedh84aeQQntu4433guY1lnDY6cR84fkQRc9sSoi+XlDN+QF9ys7PIDjCjaCCrqmrJz85icNtY7NkBjtynkDU13h+kTJFqi9RPAgcBr8YYPx9CGAbcmb6w9hz7H3Ygb76yhBvPvZreuTl89Otntu+7+Us/5oJfXExjfQN3X3UrzU3NxBgZO31/Dj3tqPZyi5+Z5yRTe7AJhx7I8jmvcdP/+z69c3N26JZ/25f/m/Nu/DaN9Q3cd3VbHWiNjJm+Pwe31YEnfvMgxSvXQwgUDB3EqV86492eShmq6KCplC5czLMXX0Z2bg5Tzz2nfd/zl13DkVd/D4Bl9zzAxhdfoaWxkae+/h1GHnsU+33swxRNO5Atry3huUuuJGRlccC/f5ycfv2663L0L+g3ZRo1ry1i+ZWXkJWTw4jPvN0tf8UPrmLCJVcAMPyMz7Dhd7+mtamJfgdOpd+UaQCU/Ol+6te/BUDvwUUM/4/Pdv1F6F29XFrO4UWF3H7swTS0tPI/i5a377v5yIO44PkFANy4ZCXfmrYfudlZvFJawcubE60Q7165jstmTOTUkcMormvgmgXLdnve0f3y+c9J44gRQoD7Vq1ndU0tABdOHsf4/onxc+9c/hbra+u77P9BOyuYNo2KxYtZcOmlZOXkMP5zn2vft+j732fa5ZcDMPbMM1n529/S2thIwdSpDJw6FYAtr7xC8ZNPAlB48MEMOSrx3qC5upqlP/85hEBOQQETvvCFrr0wpaQlwk9eXcENx04lK8DDq4pZWVXL+VNG83pZDc9uLOOhVZu4atZEHjj1EKoam7n0xaVAYoK6G46dQmtMtFa74uU32s/75Wljef/oIeRlZ/HwBw/joVXF3LpkbXddpjrJ7Td+mWOOmExRYX+Wv/QLrr7ufm6/56nuDkvvUUuE6xes4LqjppINPLKmmFXVtZw3eTRLy2t4blMZj6zZxGWHTuSeUxL3gSteSdwHqptauHv5en51/EFEEi1SXygupzC3N/99xIGJCScDzCut5M+rNnbrdUp6W4gpdAkLIbwcY5wVQphLokVqNfB6jHFSkkMB+P2Kv9vvrIdranWUj55u7uac5IW013pra8ojyWgvVdPs+I493fD85u4OQd1saUl2d4egbrbom//b3SGomx388y92dwjKALM/drQJgt14a+vDPSqHNqrvh/eo+pDqJ9s5IYQC4FZgLlADvJC2qCRJkiRJkiQpg6SUSI0xXti2eHMI4e/AgBjjwvSFJUmSJEmSJEmZI6VEagjh2F1tizE+0/khSZIkSZIkSVJmSbVr/0XbLecBs0h08T+x0yOSJEmSJEmSpAyTatf+D2+/HkIYBfwsLRFJkiRJkiRJPdAeNfNSD/SvTqG7DpjcmYFIkiRJkiRJUqZKdYzUG4HYtpoFzADmpSsoSZIkSZIkScokqY6ROme75WbgDzHG2WmIR5IkSZIkSZIyTqpjpN6e7kAkSZIkSZIkKVOl2rX/KOBKYEzbMQGIMcbx6QtNkiRJkiRJ6jmynG0qo6Xatf9XwNeBuUBL+sKRJEmSJEmSpMyTaiK1Msb4t7RGIkmSJEmSJEkZKtVE6pMhhJ8AfwQatm2MMc5LS1SSJEmSJEmSlEFSTaQe3vbz0O22ReDEzg1HkiRJkiRJ6pkcIjWzpZRIjTGekO5AJEmSJEmSJClTpdoilRDCB4EpQN62bTHG76cjKEmSJEmSJEnKJFmpFAoh3AycAXyZRCvjTwFj0hiXJEmSJEmSJGWMlBKpwJExxrOB8hjjVcARwAHpC0uSJEmSJEmSMkeqXfvr2n7WhhBGAFuA4ekJSZIkSZIkSep5QojdHYJ2I9VE6iMhhALgJ8A8IAK3pS0qSZIkSZIkScogKSVSY4xXty0+EEJ4BMiLMVamLyxJkiRJkiRJyhy7TaSGED6+m33EGP/Y+SFJkiRJkiRJUmZJ1iL1fmB+2wMgbLcvAiZSJUmSJEmSJO31kiVSPw58GpgOPAj8Ica4PO1RSZIkSZIkST1MSF5E3ShrdztjjH+OMX4aOA5YAfw0hPBcCOG4LolOkiRJkiRJkjLAbhOp26kHKoEqoB+Ql7aIJEmSJEmSJCnDJJts6kQSXftnAY8BP48xzumKwCRJkiRJkiQpUyQbI/UxYCHwHJALnB1COHvbzhjjV9IYmyRJkiRJktRjBAdJzWjJEqmf75IoJEmSJEmSJCmD7TaRGmO8PZWThBBujDF+uXNCkiRJkiRJkqTMkupkU8kc1UnnkSRJkiRJkqSM01mJVEmSJEmSJEnaayUbI1WSJEmSJElSF3CuqczWWS1S/T1LkiRJkiRJ2mt1ViL15510HkmSJEmSJEnKOCl17Q8hHABcBIzZ/pgY44ltP3+bjuAkSZIkSZIkKROkOkbqfcDNwK1AS/rCkSRJkiRJkqTMk2oitTnGeFNaI5EkSZIkSZJ6sM4ag1Ppkerv5+EQwoUhhOEhhEHbHmmNTJIkSZIkSZIyRKotUs9p+3nRdtsiML5zw5EkSZIkSZKkzJNSIjXGOC7dgUiSJEmSJElSpkopkRpC6A38F3Bs26angP+LMTalKS5JkiRJkiSpRwmhuyPQ7qTatf8moDfwy7b1z7ZtOy8dQUmSJEmSJElSJkk1kXpYjPGg7dafCCEsSEdAkiRJkiRJkpRpslIs1xJCmLBtJYQwHmhJT0iSJEmSJEmSlFlSbZF6EfBkCGElEIAxwOfTFpUkSZIkSZIkZZCUEqkxxsdDCPsDE9s2LYsxNqQvLEmSJEmSJKmncbapTLbbRGoI4cQY4xMhhI+/Y9d+IQRijH9MY2ySJEmSJEmSlBGStUg9DngC+PAu9kXARKokSZIkSZKkvd5uE6kxxivafjoeqiRJkiRJkqQeKyuVQiGEr4YQBoSE20II80II7093cJIkSZIkSZKUCVJKpAJfiDFWAe8HBgOfBX6UtqgkSZIkSZKkHib0sH97mlQTqduu7DTgjhjjaziNmCRJkiRJkqRuEEIYFEL4Zwjhzbafhbsoc0IIYf52j/oQwult+34bQli13b4ZyZ4z1UTq3BDCoyQSqf8IIfQHWjtycZIkSZIkSZLUSb4DPB5j3B94vG19BzHGJ2OMM2KMM4ATgVrg0e2KXLRtf4xxfrIn3O1kU9s5F5gBrIwx1oYQBgFOQCVJkiRJkiSpO3wUOL5t+XbgKeDbuyn/SeBvMcbaf/UJU02kHgHMjzFuDSF8BjgY+Pm/+qSSJEmSJEmSdhRCqp3HBQyLMW5sW94EDEtS/tPAde/Ydm0I4XLaWrTGGBt2d4JUfzs3AbUhhIOAbwIrgDtSPFaSJEn/v737jpOrrBc//vluOkkgFYzSey8hIFUpYruIetWLKCqIV/2JBbFcFUFE7rWLDQsWmlxEQBTRCyjSi4GEQAi9tySkkULKpnx/f5yzySTZ7MwkO9nJ7uf9eu1rZ8555pzvzDxzyvc8z3MkSZIkrSQiPhoR91T8fXSV+f+IiAfa+Xt7ZbnMTCA7WM8oYA/guorJXwZ2BvYDhtFxa1ag9hapSzIzyyB/mpm/iYiTanytJEmSJEmSJK0kM88Dzutg/hvWNC8ipkbEqMycXCZKX+pgVf8BXJWZiyuW3daadVFEnA98vlq8tbZInRsRXwaOB/4aRTvjPjW+VpIkSZIkSZI609XAh8rHHwL+3EHZ44BLKyeUyVciIoB3AA9UW2GtidRjgUXASZk5Bdgc+G6Nr5UkSZIkSZKkzvQt4KiIeAx4Q/mciBgTEb9uKxQRWwNbADev8vpLImIiMBEYAZxdbYU1de0vk6c/qHj+LI6RKkmSJEmSJHWi6OoANhiZOQM4sp3p9wAfqXj+NPCadsodUe86O0ykRsRtmXlIRMxl5QFbo1hfblzvCiVJkiRJkiRpQ9NhIjUzDyn/D16Xlew+dMm6vFzdQOsyr6j0dDe82L+rQ1AXeqajIb/VIwxapyMJdQdHbrewq0NQF7vl78u6OgR1sdE/OrmrQ1AXG/+Zc7s6BDWDdx7S1RFIa62mrv0AETGUYjyB5a/JzPGNCEqSJEmSJEmSmklNidSI+AZwAvAk0HYpOYG6xxKQJEmSJEmSpA1NrS1S/wPYLjNbGxmMJEmSJEmS1FOFN5tqai01lnsAGNLIQCRJkiRJkiSpWdXaIvWbwL0R8QCwqG1iZh7TkKgkSZIkSZIkqYnUmki9EPg2MJEVY6RKkiRJkiRJUo9QayJ1fmb+uKGRSJIkSZIkST2aY6Q2s1oTqbdGxDeBq1m5a//4hkQlSZIkSZIkSU2k1kTqPuX/AyqmJXBE54YjSZIkSZIkSc2npkRqZh7e6EAkSZIkSZIkqVm11FIoIj4TERtH4dcRMT4i3tjo4CRJkiRJkiSpGdTatf/DmfmjiHgTMBz4AHAxcH3DIpMkSZIkSZJ6kIia2jyqi9T67bTdMuytwEWZOQlvIyZJkiRJkiSph6g1kTouIq6nSKReFxGDgWWNC0uSJEmSJEmSmketXftPAvYGnszM+RExHDixcWFJkiRJkiRJUvOoKZGamcsiYiqwa0TUmnyVJEmSJEmSpG6hpqRoRHwbOBZ4EFhaTk7glgbFJUmSJEmSJPUw3pKomdXauvQdwE6ZuaiRwUiSJEmSJElSM6r1ZlNPAn0aGYgkSZIkSZIkNataW6TOByZExA3A8lapmfnphkQlSZIkSZIkSU2k1kTq1eWfJEmSJEmSpAYIx0htajUlUjPzwkYHIkmSJEmSJEnNqqZEakTsAHwT2BXo3zY9M7dtUFySJEmSJEmS1DRqvdnU+cDPgSXA4cBFwO8aFZQkSZIkSZIkNZNaE6kDMvMGIDLzmcw8E/i3xoUlSZIkSZIkSc2j1ptNLYqIFuCxiPgk8AIwqHFhSZIkSZIkST2LN5tqbrW2SP0MsBHwaWBf4HjgQ40KSpIkSZIkSZKaSdUWqRHRCzg2Mz8PzANObHhUkiRJkiRJktREOmyRGhG9M3MpcMh6ikeSJEmSJEmSmk61FqljgdHAvRFxNXA58ErbzMz8YwNjkyRJkiRJkqSmUOvNpvoDM4AjgASi/G8iVZIkSZIkSeoUtd7OSF2hWiJ104g4FXiAFQnUNtmwqCRJkiRJkiSpiVRLpPYCBrFyArWNiVRJkiRJkiRJPUK1ROrkzDxrvUQiSZIkSZIkSU2q2sAL7bVElSRJkiRJkqQepVqL1CPXSxSSJEmSJElSDxdhm8Zm1mGL1Mycub4CkSRJkiRJkqRmVa1rvyRJkiRJkiT1eCZSJUmSJEmSJKmKamOkSpIkSZIkSVovHCO1mdkiVZIkSZIkSZKqMJEqSZIkSZIkSVWYSJUkSZIkSZKkKkykSpIkSZIkSVIV3mxKkiRJkiRJagLhzaaami1SJUmSJEIgHncAACAASURBVEmSJKkKE6mSJEmSJEmSVIWJVEmSJEmSJEmqwkSqJEmSJEmSJFXhzaYkSZIkSZKkpmCbx2bmtyNJkiRJkiRJVZhIlSRJkiRJkqQqTKRKkiRJkiRJUhWOkSpJkiRJkiQ1gSC6OgR1wBapkiRJkiRJklSFiVRJkiRJkiRJqsJEqiRJkiRJkiRVYSJVkiRJkiRJkqrwZlOSJEmSJElSE4jwZlPNzBapkiRJkiRJklSFiVRJkiRJkiRJqsKu/esoMzn/nD8x/o6H6Ne/Lyef/l623Wnz1cp97RM/Y9aMOfTt1weA03/4UTYZNpgb/zqWi396DcNGbgLAW959MEcec8B6fQ9aN5nJRT+8ivvufIi+/fvysdOOY5t26sDZnzyXl6fPoU9ZB770w4+xydDBPDThCX73oz/x7BOT+eTXP8BrD99rfb8FdaLM5JnLLuPliRNp6duX7U44gYFbbbVaueeuuorpd93Fkvnz2e8nP+mCSNWZDho1lM/vuy29IrjqiSlc8ODzK80fPXJjPrfvduwwZCBfvv1hbnhuOgA7DhnIV/bfnoG9e7Es4TeTnuX6Z6d3xVvQOtp/5BA+vfu2tAT89dmpXPL4CyvN79MSnLb3juw4ZCBzWpdw5rhHmLJg0fL5mw7oy0WHjeaCR57l90++uL7DVyfITP5+3pU8cc+D9O7Xl7ed8n5etf0Wq5X7/Rk/Y97MOSxbtowtdt2ON/2/99DSq4VbLvkbE667k402GQTAYR88mu332219vw2tg9fvMIIz3roLvVqCy8Y9z89veXKl+e/fbws+8NqtWJbJK61L+PKfJvH4tHkA7LzZYP7n7bsxqF9vliW8/Rd3sGjJsq54G6rTazcdwil7bktLBH95Ziq/e3TlY4A+LcHp++7ITkMGMbt1CWfc/TBT5hfb/+023ogv7rPiOOAjN02gdVny/YN2Y3j/vvQOuG/GHL4/4QmsDRu+X3z3Y7zlyH2YNmMOY476YleHI2ktmUhdR/fe+TCTn5vOTy7/Mo9NepZffedKvvmbz7Rb9jNnvp/tdln9gPqgI/fmI5//90aHqga5786HmPL8dL5/2Vd4fNIznP+9KzjrV6e0W/YTXzuebVepAyM2G8rHTjuOv15603qIVo02+4EHWDh1KnudfTbznnqKpy65hN2/8pXVyg3Zay82O/xw7jv99C6IUp2pJeC/xmzHJ/75AFMXLOJ3b9qbm5+fyVNz5i8vM3n+Is686xE+sMvKF1kWLl3G6Xc+wnNzFzJiQF8uefM+3DF5FvMWL13fb0ProAX47B7bcupdk5i2oJXzDt2L26bM5Jl5C5aX+bctNmPu4iW875/jOeLVI/j4Lltz5vhHls//5K7b8K+XZnVB9OosT9zzIDNfnMbHzzudFx95mmt/9gdO+MHnViv3zi+dSL+NBpCZ/PGbv+Wh2+5lt9fvC8D+7ziMA/79yPUdujpBS8BZb9uN488fy5Q5C7n64wfx94deWp4oBfjz/ZO55O7nAHjDzpty+lt25kMX3UOvluCc9+zJqVfcz0NT5jJkQB8WLzVttiFoAT6313accvsDvLSglV8fvje3TZ7B03NXbP+P3qrY/h/793Ec+ZoRfGK3rTnj7kfoFXDGmJ34xj2P8vicV9i4b2+WLEsATh/7MPOXFMcC/73/zhz+mhHc8IIXWjd0F19+M7+48Dp+fc4nujoUSevArv3r6O5bHuD1b9mXiGDH3bfilXkLmDV9TleHpfVo3G0PcOibxxAR7LD71syfW18dGDlqGFtu/2oHlO4mZk2YwIgDDyQiGLzttixdsIDWl19erdzgbbel75AhXRChOtvuwwfz/LyFvPDKQpYsS657ZhqHbT5spTKTX1nEYy/Ppzw/Wu7ZuQt4bu5CAKYvaGXWwlaG9u+zvkJXJ9ll6GBeeGUhk+cvYkkmN7w4jUNetXIdOORVw7j2+ZcAuHnydEaXPVHa5k2ev4in585HG65H/zWRPY7Yn4jgNTtvw8JXFjBv5uzVyvXbaAAAy5YuY+niJe7/u4m9Nx/CMzNe4blZC1i8NPnLxMm8cZdNVyozb9GS5Y836tuLtl3CoduP4OEpc3loylwAXl6weLX9hZrTLsMG8/wrC3mxbfv//DQOHTV8pTKHjhrO354ttv83vTidfUcWx3/7bzqUJ2a/wuNzXgFgTuuS5a1O25KovSLo3eIpe3dx+9iHmfnyvOoFJaKH/W1YqrZIjYjRHc3PzPGdF86GZ+a02QzfbEUyZPjITZg5bTZDR2y8Wtlzz/49Lb1aOOCwPXnXiW9YfuD8r5vu56EJTzJqy5Gc8JljGLHZ0PUWv9bdzGlzGL7pijowbNMhzFpDHfjl/1xKS0sL+x+2J+844ShPnrqh1pdfpt/QFb/hvkOH0vryyyZNu7GRA/ox5ZUVXbRfmt/K7iMG172c3YYPok9LC8+XiVVtOEb078tLC1qXP5+2sJVdhwxup0xRT5YmvLJ4CZv07U3r0mW8b7vX8Lm7JvHe7V6zXuNW55o3YzYbj1ixrR88fAhzZ8xm0LBNVit76ek/Y/Kjz7DtmF3Z+eC9l08fd82tTPzn3YzafguO/Mg7GTBoo/USu9bdZhv358XZK7bfk+csZO/NV9/3f+C1W/KRg7ehT6/gfb8dC8C2wweSwEUfGsOwgX35y/2T+eVtT62v0LUORlZs2wFeWrCI3YauvP0fOaAvL81fffu/xaABJPCDg3ZjSL8+/OP5afzvYyuGhfnBQbuxy9DB3DV1JjfaGlWSmkYtXfu/38G8BI7opFi6tU+f+X6Gb7oJC15ZyPe+ciG3/N84Xv/WMYw5ZDcOOWo0ffr25u9X3clPv/F7zvzp/+vqcNUAn/ja+xk2cggLXlnID0+7gNuuvYdD37JfV4clqQmM6N+Hbxy4E1+781FshNSznLjTllz+5IsssBtvj3LcNz7BktbF/Pl7F/HM/Y+yzT47M/qth3DIe99MBNz8u79xw6+v4uhT3t/VoaqTXfyvZ7n4X89yzJ6j+NRh2/G5KyfSqyXYb6uhHPPzO1iweCn/e+L+THxxDnc8OaOrw1UD9Ypgz+Eb85GbJrBw6TJ+fMjuPPLyPMZNK1qyn3rHJPq2BF8bsxP7jhzC3dNW7+EkSVr/qiZSM/PwtVlwRHwU+CjA6T84mXd/6M1rs5imdO0Vt/GPq/8FwPa7bMGMqSt2ajOmzV5+46hKwzctpg0Y2J9D3rgPjz34LK9/6xgGbzJweZkjjnktF597TYOjV2e4/srbuPHquwDYdpctmPHSijow86WXGdpOHRhWduMZMLA/Bx01micefNZEajcx5cYbmXbrrQAM3HprFs2aRVtbhNZZs2yN2s1NW7CIVw3st/z5phutaHlSi4G9e/Gjw3bn3PueYeKMuY0IUQ02fWErmw7ou/z5yP59mbZwUTtl+jFtYSu9Agb26c3s1iXsMmQQrx81nI/vujWD+vQmM2ldtow/Pj1lfb8NrYV7rrmFCdfdCcCrd9iSOdNXHA/MnfEyg4evfjzQpnffPuz42j149K6JbLPPzgwauqIny95vOpA/fP28xgWuTjd1zkJevUn/5c9HbdyfqXPW3MPgLxMnc/YxuwETmTJnIWOfnsms+YsBuPHRaez+6o1NpG4AppXb9jZt2/mVyixoZdONVt/+v7RgEffNmM3s1mLIhzunzGKnIYOWJ1IBWpclt06eyaGjhplIlaQmUdfNpiJid2BXYPlRQmZe1F7ZzDwPOA/g/pnXdKsGNm9+9yG8+d2HADDu9ge59orbOfiofXhs0rNsNLD/al26ly5ZyivzFrDxkEEsWbKUcbc/xJ5jdgBg1vQ5y8vfc+skNt965bGU1Jze+K5DeOO7ijpw7x0Pcv2Vt3HgG/bh8UnPMGBQ+3Vg/rwFDC7rwL13PMjuY3bsitDVAK86/HBedXhxzWnW/fcz9cYbGb7ffsx76il6DRhgIrWbmzRjLlsM7s+rB/bjpQWtvGmrkXzljkeqvxDo3RJ8/3W78tenpnLDc3bb21A9/PJcNh84gFHlCfSRrx7JWeNXrgO3T53JmzfflEmz5vL6USMYP704Uf7UHQ8sL3PijluwYMlSk6gbkDFHv44xR78OgMfvnsQ919zCrq8bzYuPPE2/jfqv1q2/dcEiWhcsZNCwTVi2dCmP3zOJLXbdDoB5M1cMA/DonfczcqtR6/fNaJ3c98Jsth4+kM2HDmDqnIW8bY9RfPry+1Yqs/XwjXh6RjEW8hE7brr88c2PTeNjh25D/z4tLF6avHabYfzm9qfX91vQWnh41lw2HzSAURv1Y9qCVo7cfCRfv3vl7f9tk2fy1i03ZdLMuRz26hGMKxOiY1+axft33Jx+vVpYsmwZe4/YhMsef4EBvVrYqHcvZixaTK+Ag141lPtmeA8OqScJb2fU1GpOpEbE14DDKBKpfwPeAtwGtJtI7SlGH7QL997xEJ96zzfp268PJ3/1vcvnff6D3+d7F32OxYuXcPYpv2LpkqUsW7aMPfbbkSPffgAAf/vDrdxz2yR69Wph0MYbrfR6bRj2PnAXJtz5EKf+x//Qt38fPvaV45bP+/KHvsc3L/w8ixcv4VunnlfUgaXL2H2/HTnimKIOPPHQs5zz5fOZP3cB994+iSt/fS3fueS/uurtaB0N2WMPXn7gAe477TRa+vZl2xNOWD5v4llnsccZZwDw7BVXMH3sWJa1tjL+i19k00MOYfNjjumiqLUuliZ8+54nOPfw3WmJ4Oonp/Lk7Pl8fI+teHDmXG55YSa7DhvE91+3Kxv37c3rXjOMj++xJe/523jeuOUI9tl0Yzbp15u3bbsZAF+781EeffmVLn5XqsfShB8+8CTfO2A3WgL+9txLPD1vAR/eaUseeXket0+dyV+fncpp++zI/x4xmrmtSzhzfG3Jdm04thuzK4/fM4mf/+dZ9OnXd6Vu+b/+1Lf5yE/+i9aFi7j8G79iyeIl5LJkqz13YPRbDwbgn+f/malPvgARDNl0GG/55LFd9Va0FpYuS8645kEu+tB+9GoJ/jDueR57aR6fPXIHJr4wm388/BIfeu1WHLzdcJYsS2YvWMznrrwfgDkLl/Dr25/m6o8fRFK0SL3x0Wld+4ZUk6UJ59z3BD84eHd6Adc8M5Wn5s7nI7tsycOz5nHblJlc88wUTh+zE5cdtS9zWpfwtbsfBmDu4qX8/vEX+M1he5EULVLvnDqLof368O0Dd6VPSwstAeOnzeZPT03u0vepznHhTz7FoQfuwoihg3n8Xz/lGz+4ggsvu6mrw5JUp8isrbFoREwE9gLuzcy9ImIz4HeZeVS113a3FqmqX+syb6rU0/140qCuDkFd6IEXvKra0w0abB3o6U7a2TsV93Rfu9CxgHu61+xb/80Y1b2M/8y5XR2CmsCCZy81QdCBRUvH9qgcWr9e+29Q9aGes5oFmbkMWBIRGwMvAVs0JixJkiRJkiRJah71jJF6T0QMAX4FjAPmAXc2JCpJkiRJkiRJaiI1J1Iz8xPlw19ExLXAxpl5f2PCkiRJkiRJknqaDaqne49Tz82mXtfetMy8pXNDkiRJkiRJkqTmUk/X/i9UPO4P7E/Rxf+ITo1IkiRJkiRJkppMPV3731b5PCK2AH7Y6RFJkiRJkiRJUpNpWYfXPg/s0lmBSJIkSZIkSVKzqmeM1J8AWT5tAfYGxjciKEmSJEmSJKmnifBmU82snjFS76l4vAS4NDNv7+R4JEmSJEmSJKnp1DNG6oWNDESSJEmSJEmSmlXVRGpETGRFl/7VZOaenRqRJEmSJEmSJDWZWlqkHl3+P7n8f3H5/3g6SLBKkiRJkiRJqodjpDazqonUzHwGICKOysx9Kmb9V0SMB77UqOAkSZIkSZIkqRm01FE2IuLgiicH1fl6SZIkSZIkSdog1XyzKeAk4LcRsQlFO+NZwIcbEpUkSZIkSZIkNZGaE6mZOQ7Yq0ykkpmzGxaVJEmSJEmSJDWRqonUiDg+M38XEaeuMh2AzPxBg2KTJEmSJEmSeoxwFM2mVkuL1IHl/8GNDESSJEmSJEmSmlXVRGpm/rJ8+LPMnNbgeCRJkiRJkiSp6dTTXvj2iLg+Ik6KiKENi0iSJEmSJEmSmkzNidTM3BH4KrAbMC4iromI4xsWmSRJkiRJkiQ1ibpGsM3MsZl5KrA/MBO4sCFRSZIkSZIkST1O9LC/DUvNidSI2DgiPhQR/wfcAUymSKhKkiRJkiRJUrdW9WZTFe4D/gSclZl3NigeSZIkSZIkSWo69SRSt83MbFgkkiRJkiRJktSkqiZSI+KHmXkKcHVErJZIzcxjGhKZJEmSJEmS1IPEBjhuaE9SS4vUi8v/32tkIJIkSZIkSZLUrKomUjNzXPn/5saHI0mSJEmSJEnNp5au/ROBNY6Nmpl7dmpEkiRJkiRJktRkaunaf3T5/+Tyf1tX/+PpIMEqSZIkSZIkSd1FLV37nwGIiKMyc5+KWf8VEeOBLzUqOEmSJEmSJKmniPBmU82spY6yEREHVzw5qM7XS5IkSZIkSdIGqZau/W1OAn4bEZsAAcwCPtyQqCRJkiRJkiSpidScSM3MccBeZSKVzJzdsKgkSZIkSZIkqYnUnEiNiH7Au4Ctgd5tYzZk5lkNiUySJEmSJEmSmkQ9Xfv/DMwGxgGLGhOOJEmSJEmS1FN5O6JmVk8idfPMfHPDIpEkSZIkSZKkJlVPmvuOiNijYZFIkiRJkiRJUpOqp0XqIcAJEfEURdf+ADIz92xIZJIkSZIkSZLUJOpJpL6lYVFIkiRJkiRJPVwQXR2COlA1kRoRw8qHcxsciyRJkiRJkiQ1pVpapI4DEtpNiSewbadGJEmSJEmSJElNpmoiNTO3qWVBEbFbZk5a95AkSZIkSZIkqbm0dOKyLu7EZUmSJEmSJElS06jnZlPVOBquJEmSJEmStNZMrzWzzmyRmp24LEmSJEmSJElqGp2ZSJUkSZIkSZKkbqkzE6mtnbgsSZIkSZIkSWoaNSdSo3B8RJxRPt8yIvZvm5+ZBzQiQEmSJEmSJEnqavXcbOpnwDLgCOAsYC5wJbBfA+KSJEmSJEmSepQIbzbVzOpJpL42M0dHxL0AmTkrIvo2KC5JkiRJkiRJahr1jJG6OCJ6AQkQESMpWqhKkiRJkiRJUrdWTyL1x8BVwGYR8d/AbcD/NCQqSZIkSZIkSWoiNXftz8xLImIccGQ56R2Z+VBjwpIkSZIkSZJ6mnraPGp9q2eMVICNgLbu/QM6PxxJkiRJkiRJaj41p7kj4gzgQmAYMAI4PyK+2qjAJEmSJEmSJKlZ1NMi9f3AXpm5ECAivgVMAM5uRGCSJEmSJEmS1CzqGXjhRaB/xfN+wAudG44kSZIkSZIkNZ96WqTOBiZFxN8pxkg9ChgbET8GyMxPNyA+SZIkSZIkqUcIoqtDUAfqSaReVf61ualzQ5EkSZIkSZKk5lRPInUm8NfMXNaoYCRJkiRJkiSpGdUzRuqxwGMR8Z2I2LlRAUmSJEmSJElSs4nMrL1wxMbAccCJFOOkng9cmplzGxNe9xERH83M87o6DnUd64CsAz2b37+sA7IOyDog64CsA9KGrZ4WqWTmHOAK4PfAKOCdwPiI+FQDYutuPtrVAajLWQdkHejZ/P5lHZB1QNYBWQdkHZA2YFUTqRHx7+X/YyLiKoqbTPUB9s/MtwB7AZ9rZJCSJEmSJEmS1JVqudnUV4E/Au8CzsnMWypnZub8iDipEcFJkiRJkiRJUjOoJZEKQGZ+qIN5N3ROON2aY6DIOiDrQM/m9y/rgKwDsg7IOiDrgLQBq3qzqYiYDzze3iwgM3PPRgQmSZIkSZIkSc2ilhapTwFva3QgkiRJkiRJktSsakmktmbmMw2PRJKkDUBEbA6cC+xKcdPGa4AvZGZrlwYmSZIkSWqolhrK3F7LgiJijWOoNpOIOC0iJkXE/RExISJe20HZCyLi3eXjmyJiTPn4bxExpBNjOiwiZpfxPBQRX1tDuTER8ePOWm9PEhFLy8/3gYi4PCI26oIYDouIg6qUOTMiXqiI9Zg1lPt4RHywMZH2DBExr+LxWyPi0YjYqgHruSAinoqI+8p1XFQm4tZU/tcRsWtnx6HOERFBcQPGP2XmDsCOwCDgv9fDumse11wdi4gbI+JNq0w7JSJ+vobyT0fEiHamHxMRX6qyrnkdza/y2ndEREbEzmu7jJ6kYl/f9rf1WizjsIi4Zg3zTqxYdmtETCwff2tdY+8gng73IQ04Jj0zIj7fWcvrCt28HkyIiPERcWCV8l9Zi3VsHREPrH2UjRERwys+7ykVx8kTIqJvF8V0x1q85qyIeEMnxnBCuX94Q8W0tn3GuztrPTXEsfwcuYHrWOfz93VY97nlOh+MiAUVda9hn/Gajjk6KL/8txs15AvWdB65rtuA9VEXpGZR9aQsMz9Z47I+A1y4buE0VnnQcTQwOjMXlRuounfAmfnWTg8Obs3MoyNiIDAhIv6SmePbZkZE78y8B7inAevuCRZk5t4AEXEJ8HHgB20zy893SYNjOAyYB1Q7+DonM78XEbsAt0bEppm5rG1mGesvGhhnjxIRRwI/Bt7UwNb3X8jMK8ok3CnAPyNi91VbMEZEr8z8SINiUOc4AliYmecDZObSiPgs8FR58PiZzLw/Iu4FrsrMsyLiLOA54DHgTGA6sDswDjg+MzMi9qXYJg0q55+QmZMj4iZgAnAIcCnw/fX4XruzS4H3AtdVTHsv8MV6FpKZVwNXd2JcqzoOuK383+5FVq1k+b6+Ecrf/flQnOgCh2fm9Eatr8Ia9yENOibd0HX3evBG4JdAR/ep+ArwP6tOLOtQVB5XNrvMnAG0HcOfCczLzO+1zV9Px/CrxtRhw4g1vOaMBoQykWLf9Y/y+XHAfQ1YT5fprPP3GtfVKzOXVk7LzJPLeVsD1zRy29IZaskXeB4prbtaWqTWKjpxWY0yCpiemYsAMnN6Zr4YEftGxM0RMS4irouIUR0tpO0qUXnV5qGI+FV5lez6iBhQltmv4qrZd2u9upOZr1CcXG9ftgq4OCJuBy6uvDoeEYMi4vzyKvj9EfGucvobI+LO8mr15RExaO0/rm7rVorP97CIuDUirgYejIhe5Xd1d/mZfgwgIkZFxC2xopXooeX0dj/rsn58vZw+MSJ2Lne+Hwc+Wy7n0GpBZuZDwBJgRHmF74cRcQ/wmahoMRIR20fEP6JorTI+IrYrp3+h4r18vdM/xW4gIl4H/Ao4OjOfKKddEBE/j4i7IuLJsp78tvytX1CW6VWWe6D8jj9by/qycA4wBXhLuax5EfH9iLgPOLD8rsdEcbX4uxWxnhARPy0fHx8RY8u69MuI6FWxrP8u68JdEbFZ531aKu1GsY1eLjPnAM8CNwKHRsQmFL/dg8sihwK3lI/3oUiE7ApsCxwcEX2AnwDvzsx9gd+ycgvXvpk5JjNNonaeK4B/i7I1U7mNfjUwoIN96Kcqt+vl6yp/l5tFxFXl7+++aKcHQj3b5XLdhwAnUZwot01viYifRcTDEfH3KFoktrW+qet4pieIipY95bb1pvLxwHLbPjYi7o2It6/l8j8cET+seP6fEXFOFMeID0fEJeX+44ooe8Os7fe0hn1I5ftbbd+wpv1VRGwXEdeWMdwa3bzVc3eqBxT7k+3LZbT3nX+LYls2oVzv1hHxSERcBDwAbBHluUlZJ45dm/fclco6/YuI+BfwnYjYv9x23xsRd0TETmW5EyLij2VdfywivlNOX9Pv4qbye7un/L72K1//WEScXbH+eeX/1c4ROlh2ZUvJI8tYJ5b1r185fbVziCofxa3A/hHRJ4p9xvYUF1/b4jwjin3OAxFxXkRExfv8dll3Ho0V5zbL92nl82si4rDy8c/Lz2VSrN/zis46f+/ovO3bETEeeE8tAUXRM+AdFc8viYi3l5/fn8vP97Go6GXa3m+1hvV0lGvYN8rjDeDkitccVn5vLeV7G1Ix77EojlUqzyPXtJxmrAtS0+jMRGp24rIa5XqKg4dHozgJeX1UP4GtZgfg3MzcDXgZeFc5/XzgY+VVq6VrevGqImI4cAAwqZy0K/CGzDxulaKnA7Mzc4/M3JOidcII4Ktl+dEUV6NOreO9dHtRdI19C8UVXIDRFC3IdqQ4WZ2dmfsB+wH/GRHbAO8Driu/y70oWgxX+6ynl9N/Dnw+M58GfkHR2nTvzLy1hlhfCywDppWT1pRMuYSiDu4FHARMjqK1wg7A/hRX8feNImmoFfoBfwLekZkPrzJvKHAg8FmK1mbnUCTQ9oiIvSk+09dk5u6ZuQdly5Q6jAfaDo4HAv/KzL0y87aKMlcC76x4fizw+yhaKh8LHFyxfXl/xbLuKuvCLcB/1hmX1s3NwOsoEqh/BQZFccK8TWY+UpYZm5nPl62BJgBbAztRtFD9e0RMoNi2VA7/cNl6ir/HyMyZwFjKZBRFovJ64DRq3K63s9gfAzeXv7/RrNiPA8VJHPVtl98OXJuZjwIzomi1DPDvFPVmV+ADFNsqOuF4pjtoSx5NiIirqpQ9DfhnZu4PHA58N4peQfX6A/C28vMHOJHis4fit/2zzNwFmAN8opO+p8p9CAAd7BvWtL86D/hUGcPngZ/VGUMz6+714G3AxDV955n5JcpWuZnZdnywQxnDbsAYinqxF/CG8j1viBddNgcOysxTgYeBQzNzH+AMVm6NuzfF57QHcGxEbEHHx3GtmTmG4rj9zxTJpd2BE8rztEqrnSNUWTYR0R+4ADi2nN8b+H8VRartayolRWvUN1HsM1btIfHTzNwvM3cHBlC07GzTu6z3p1Bbj4fTys9lT+D1EdFRi+jOtM7n7zWct83IzNGZ+fsaY/oNcEK57E0ozr/+Ws7bnyIfsCfwnigu3HR07F5NR7mGT5XHHKspjzP/THkuUZ5XPpOZl8hlUwAADMlJREFUU1cp2uFy1qCr6oLUNDpzvLWmb5GamfPKE5FDKQ6WLgPOZsUJLEAvYHIdi30qM9uu/I0Dti6v/AzOzDvL6f/Lyjuu9hwaRVfQZcC3MnNSRLwHuDozF7RT/g1UtFDJzFkRcTTFidXt5XvpC9zZzmt7ogFlggKKq7e/odjpjc3Mp8rpbwT2jBVj3mxCsfO6G/htudP+U2ZOiIjX0/Fn/cfy/ziKk956fDYijgfmUhxkZbmO1ZIpETGY4mDtKoDMXFhOf2P5fu4tiw4q38stqy6jB1tMMczCSRRDk1T6S/m5TwSmZuZEgIiYRJHAuBnYNiJ+QnHgdH2d667cXi6lSJquJDOnRdEi9gCKLuE7U4xZfTKwL3B3WS8GAC+VL2uluPERFHXvqDrjUnUPAiuNixURGwNbUvzexgBPAn8HRlAksytbsC6qeLyUYj8cwKTMXNOYd690SuRaVVv3/j+X/68C3sHab9ePAD4IxZAPwOxV5te7XT4O+FH5+Pfl83EUrVQvL0+SpkTEjWWZyoQ81H880x3U06X7jcAxsWI80P4Uv+O6lMeW/wSOjoiHgD6ZOTGKVs7PZWbbvQZ+B3wauJZ1/57aO+Y+kvb3DX9hlf1VFC2xDgIuL8tCcXGxu+iu9eC7EfFVigvsJ7Hm77w9z2TmXeXjQ4BLy+3U1Ii4maIBwf31vesud3mu6Ia9CXBhROxAkVzsU1HuhsycDRARDwJbUVzoWtNxXFsyciLFvnly+dongS2AGRVl2ztHeLKDZUOxrX6qvEgGxdB4JwNtLZrrPYf4PUWd2gT4HMWwDm0Oj4gvAhsBw8r3/Zd21rN1Dev5j4j4KMVxyyiK86CG15lOOn8/gI7P2+q6YJ2ZN5dJ3ZEUic0rM3NJuey/ZzEcBRHxR4rf2xJq/62uak25hiGZ2Xb8cDErLgxXuoziwsL5FMc5K73POpazqi6pC1IzqSmRGhEtFFd8/tBBsZpuStXVyh3uTcBNZZLkZDo+ga1m1ZPiAWu5nFszs71kaz0n0EGx8V619araOagud2SVn29QXJGrHDOvrezrgH8DLoiIHwCz6PizbqsXbYmSepyTFWM/Vai3LnwzM39Z57p7kmXAfwA3RMRXMrOy9cKiijKLVnlN7/LCxV4ULQA+Xi7nw3Wsex/ghvLxwlxlPKYKvy+X/TDFeJsZRcW9MDO/3E75xZnZ1jtgbeqeqrsB+FZEfDAzL4qia9b3gQsyc05EPEfRNewsYCTwvfKvI48AIyPiwMy8szwh2zEzJ1V5ndbNn4FzImI0xUnmeBq3XYc6tssRMYwiMbtHRCTFSWJGxBeqLH9djme6qyWs6IHVv2J6AO+qaC1eTFy7IVF+TZG8eJiVW5+t2lsr6ZzvqXIf0maN+4Z29lenAC/XkWzsDrpDPfhCZl5REePhrPl4YFXd8YJc5Xv6BnBjZr6zTF7fVDFvtQuYVY7jOjwGrAwgM29Z9RyhPDZYl2PEuvY1mTk2IvYA5mfmo20XR8qWrz8DxmTmc1GML1tZ99tbT+XvhLbyUfTQ+zywX/nZXbDKshqqE87fq50jr83v4yLgeIoE5YmV4a5Sru33XutvdVXrkmu4k2I4u5EUF4rPrlK+UlPWBalZ1NS1v2z10OENGLL2m1J1mYjYqbxS2WZv4CHKE9iyTJ+I2G1d1pOZLwNzY8UdBd/bUfm19HdWHsdkKHAXxXh7beMmDYyIHRuw7u7qOuD/lUkMImLH8jPciqJV4q8oDpJHs3af9VxgcGcGnJlzgeejHKcnIvpF0ZX4OuDDsWL8n9dExKadue7uIDPnUxz8vj8iTqr1dVF0EWrJzCspugqNrvF1ERGfprh6e20NL7mKoqvWcRRJVShOnt/d9n1GxLCyjmo9KBPV76TorvUY8CiwkBUtQG4FXip7EtxK0fWww6E8srjp2LuBb0cxRtUEitZiaqDMnEcxru1vKVqnrus+9AbK7plRjJG3ySrz69kuvxu4ODO3ysytM3ML4CmKFjm3A++KYvyzzShuZAgVCfly+et8PNNNPE3REghWdImE4vv4VHlxiojYZ21XkJn/omip9j6KutRmy1hxd/X3Udw4bK2/pyr7kHb3De3tr7IY1/mpKHo+tS23nm6dG6Kn6Sb1oEJHxwOLY8UwA6u6laKLe68ywfI6iqFONmSbAC+Uj0+oVnhtj+PaWc5q5wg1LPsRilaF25fPP0DR02ldfImVW6LCiuTW9HK/U8td5p8G9i73L1tQdFMH2Jgi2Ti73O/U0mqxU3TS+XsjzpEvoLgoRWY+WDH9qPK3OICylwudfOxe5hpejohDykntDhNQHrNeRXEz04faWsrWuJynabK6IDWTelpU/COKbi+XUXHVJotxxjYUg4CfRNGMfQnwOPBRinGiflye9PSm6Fqxri2BTgJ+FRHLKHaOq3bxW1dnA+dGcROrpcDXM/OPEXECcGmUg5ZT7MAfXcMytLJfU3RtGV8eUE+j2AEeBnwhIhYD84APlt2uT6C+z/ovwBVR3MjgU1nDOKk1+gDwyyjuDL4YeE9mXh/FeDx3lucG8yiumtbajaTHyMyZEfFm4JaImFb1BYXXAOdH0VofoNoV5u9GxOkUrd7uorjDb2sNsc2Koovgrpk5tpz2YBRd+64v17+Y4qLKMzXGrnWUmc9RjFHX3rzTKcawJjNfpKILbmbeREUrmcoLkGW3rdXGy8zMwzonaq3BpRQnGe9dy+16pc8A55UXZZZSJFWXdx2sc7t8HPDtVaZdWU4/maJL74PAcxQtaWdnZmsUQ9N09vHMhu7rwG8i4hus3ErtGxSfz/3ltvQpqg/D1JE/AHtn5qyKaY8AJ0fEbym+r5+v5fdUdR/Swb5hAe3vr94P/Lx8TR+Ki3Xd6m7fq+gO9WAlVY4Hzivf03iKcWArXUUxtvJ9FK3lvpiZU6Joybmh+g5F1/6vsmKsyo7Uexy3JoexyjlCtWVn5sKIOJFiaI3eFMMDrNNd1DPz/9qZ9nJE/IriBmNTyvVUczvFb+BBimTl+HJZ90UxBN3DFPud9dkTdZ3P3zth/97eMqeWx+h/WmXWWIr99ebA7zLzHoAGHLufSDGsRNLxEGOXUXz3J9S5nGasC1LTiBW9QKsUjHiqncmZmdt2bkjdQ0QMKlu7EBFfAkZl5qrjMEqSJNWl7RgjihufjKW4gcWUro6rJ4uIayiG5rmhfL41cE0WN3lRD2E9kHqGKHoATqRo5d82Bu8JFEMpNH1PXUnrpuYWqZm5TSMD6Yb+LSK+TPEZP0MN3UwkSZJqcE3ZOqcv8A2TqF2n/B7GAve1Jc/U81gPpJ4jIt5AcePic9qSqJJ6lnpapG4EnApsmZkfLccq2Skzr6nyUpUi4k2s3lXvqcx8Z1fEo64TEadR3JCm0uWZ+d9dEY/WXUScCxy8yuQfZeb57ZWXpLJFaXtJlyNXHctM61fZ9XbVnkS3Z+bJ7ZXvhPW5D2lC1gN1tfVdB3uy9f37i4h/Af1WmfyBzJzYiPVJ6jz1JFIvA8ZRjA+5e5lYvaOH3fFTkiRJkiRJUg/UUr3Icttl5ncoBkduu9t1dPwSSZIkSZIkSdrw1ZNIbY2IARR3dyQitgMWNSQqSZIkSZIkSWoiNd9sCvgacC2wRURcQjF+yAmNCEqSJEmSJEmSmknNY6TC8psiHEDRpf+uzJzeqMAkSZIkSZIkqVlUTaRGxOiO5mfm+E6NSJIkSZIkSZKaTC2J1Bs7mJ2ZeUTnhiRJkiRJkiRJzaWurv2SJEmSJEmS1BNVvdlURPx7R/Mz84+dF44kSZIkSZIkNZ+qiVTgbR3MS8BEqiRJkiRJkqRuza79kiRJkiRJklRFS60FI2KziPhNRPxf+XzXiDipcaFJkiRJkiRJUnOoOZEKXABcB7y6fP4ocEpnByRJkiRJkiRJzaaeROqIzPwDsAwgM5cASxsSlSRJkiRJkiQ1kXoSqa9ExHCKG0wREQcAsxsSlSRJkiRJkiQ1kd51lD0VuBrYLiJuB0YC725IVJIkSZIkSZLURKq2SI2I/SLiVZk5Hng98BVgEXA98HyD45MkSZIkSZKkLldL1/5fAq3l44OA04BzgVnAeQ2KS5IkSZIkSZKaRi1d+3tl5szy8bHAeZl5JXBlRExoXGiSJEmSJEmS1BxqaZHaKyLaEq5HAv+smFfPGKuSJEmSJEmStEGqJRF6KXBzREwHFgC3AkTE9sDsBsYmSZIkSZIkSU0hMrN6oYgDgFHA9Zn5SjltR2BQeRMqSZIkSZIkSeq2akqkSpIkSZIkSVJPVssYqZIkSZIkSZLUo5lIlSRJkiRJkqQqTKRKkiRJkiRJUhUmUiVJkiRJkiSpChOpkiRJkiRJklTF/wc2v5TvGVUWcAAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 1872x1584 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YlbuNdGS4xKl"
      },
      "source": [
        "#### I have skipped the EDA part as the main idea is to create the ml model.\n",
        "#### Try to do some visualizations, in order to understand the features of this dataset."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Sdi3LbHH4xKm"
      },
      "source": [
        "### Features and target variable"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EPjlFwFg4xKm"
      },
      "source": [
        "# taking all the features except \"selling price\"\n",
        "X=dataset.iloc[:,1:]\n",
        "# taking \"selling price\" as y , as it is our target variable\n",
        "y=dataset.iloc[:,0]\n"
      ],
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FDIy9uJ54xKo"
      },
      "source": [
        "### Feature Importance"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8BGA-1sN4xKp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b9fca482-bffe-4949-d4d9-a42f56b4e9e2"
      },
      "source": [
        "#checking and comparing the importance of features\n",
        "from sklearn.ensemble import ExtraTreesRegressor\n",
        "#creating object\n",
        "model = ExtraTreesRegressor()\n",
        "#fit the model\n",
        "model.fit(X,y)\n",
        "\n",
        "print(model.feature_importances_)"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[0.35597614 0.04023309 0.00044603 0.07432801 0.23860635 0.0091719\n",
            " 0.14492702 0.13631145]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s4m7ccXL4xKw",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "bab88244-e7e7-42ed-b452-17e6089c405a"
      },
      "source": [
        "#plot graph of feature importances for better visualization\n",
        "feat_importances = pd.Series(model.feature_importances_, index=X.columns)\n",
        "# considering top 5 important features\n",
        "feat_importances.nlargest(5).plot(kind='barh')\n",
        "plt.show()\n"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AGBLKN694xKz"
      },
      "source": [
        "### Splitting data into training and testing"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UC6vjou44xK4"
      },
      "source": [
        "#splitting the data\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)"
      ],
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xtn1mCvP4xK-"
      },
      "source": [
        "# Fitting and evaluating different models\n",
        "Here I am using three models :\n",
        "1. Linear Regression\n",
        "2. Decision Tree\n",
        "3. Random forest Regressor\n",
        "\n",
        "I will fit these models and then choose one with the better accuracy.\n",
        "You can use any regression model as per your choice."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LCdQf3ql4xK_"
      },
      "source": [
        "## Linear Regression Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pCZk8mCM4xLA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "41b645e2-7d37-4eb0-a9fb-0b0b7b4cbdc7"
      },
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "#creating object for linear regression\n",
        "reg=LinearRegression()\n",
        "#fitting the linear regression model\n",
        "reg.fit(X_train,y_train)\n",
        "\n",
        "# Predict on the test data: y_pred\n",
        "y_pred = reg.predict(X_test)\n",
        "\n",
        "#metrics\n",
        "from sklearn import metrics\n",
        "#print mean absolute error\n",
        "print('MAE:', metrics.mean_absolute_error(y_test, y_pred))\n",
        "#print mean squared error\n",
        "print('MSE:', metrics.mean_squared_error(y_test, y_pred))\n",
        "#print the root mean squared error\n",
        "print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))\n",
        "#print R2 metrics score\n",
        "R2 = metrics.r2_score(y_test,y_pred)\n",
        "print('R2:',R2)"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: 1.2426713915033711\n",
            "MSE: 4.432128265667619\n",
            "RMSE: 2.1052620420431323\n",
            "R2: 0.8517983059778262\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "26fpXmz84xLC"
      },
      "source": [
        "## Decision tree Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ouDbckh_4xLD",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7d4dce4f-3b60-42b9-ce17-d7efd118d69b"
      },
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\n",
        "\n",
        "#creating object for Decision tree\n",
        "tree = DecisionTreeRegressor()\n",
        "\n",
        "#fitting the decision tree model\n",
        "tree.fit(X_train,y_train)\n",
        "\n",
        "# Predict on the test data: y_pred\n",
        "y_pred = tree.predict(X_test)\n",
        "\n",
        "#print errors\n",
        "from sklearn import metrics\n",
        "print('MAE:', metrics.mean_absolute_error(y_test, y_pred))\n",
        "print('MSE:', metrics.mean_squared_error(y_test, y_pred))\n",
        "print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))\n",
        "R2 = metrics.r2_score(y_test,y_pred)\n",
        "print('R2:',R2)"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: 0.8348351648351646\n",
            "MSE: 2.47037032967033\n",
            "RMSE: 1.571741177697629\n",
            "R2: 0.9173956515303804\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U3kcYQIK4xLE"
      },
      "source": [
        "## Random Forest Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vYPzNh304xLE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0fef19ae-e9ee-4baf-8536-ba3b8c6db924"
      },
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "#creating object for Random forest regressor\n",
        "rf = RandomForestRegressor(n_estimators = 100, random_state = 42)\n",
        "\n",
        "#fitting the rf model\n",
        "rf.fit(X_train,y_train)\n",
        "\n",
        "# Predict on the test data: y_pred\n",
        "y_pred = rf.predict(X_test)\n",
        "\n",
        "#print errors\n",
        "from sklearn import metrics\n",
        "print('MAE:', metrics.mean_absolute_error(y_test, y_pred))\n",
        "print('MSE:', metrics.mean_squared_error(y_test, y_pred))\n",
        "print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))\n",
        "R2 = metrics.r2_score(y_test,y_pred)\n",
        "print('R2:',R2)"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: 0.7551296703296704\n",
            "MSE: 2.6380509257142837\n",
            "RMSE: 1.6242077840332756\n",
            "R2: 0.9117887406066014\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BI7pTJVI4xLF"
      },
      "source": [
        "#### We want our R2 score to be maximum and other errors to be minimum for better results"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pRBu16884xLG"
      },
      "source": [
        "### Random forest regressor is giving better results. therefore we will hypertune this model and then fit, predict."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_43Apa374xLG"
      },
      "source": [
        "# Hyperparamter tuning"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m365cUnD4xLH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "600eaffd-7a37-4977-c404-649c2e968439"
      },
      "source": [
        "#n_estimators = The number of trees in the forest.\n",
        "n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]\n",
        "print(n_estimators)"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9W48_Ddw4xLH"
      },
      "source": [
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "\n",
        "#Randomized Search CV\n",
        "\n",
        "# Number of trees in random forest\n",
        "n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]\n",
        "# Number of features to consider at every split\n",
        "max_features = ['auto', 'sqrt']\n",
        "# Maximum number of levels in tree\n",
        "max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]\n",
        "# max_depth.append(None)\n",
        "# Minimum number of samples required to split a node\n",
        "min_samples_split = [2, 5, 10, 15, 100]\n",
        "# Minimum number of samples required at each leaf node\n",
        "min_samples_leaf = [1, 2, 5, 10]\n"
      ],
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HI9V1CAl4xLI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7a97f86c-7201-47a7-c2b7-34da031b8484"
      },
      "source": [
        "# Create the random grid\n",
        "random_grid = {'n_estimators': n_estimators,\n",
        "               'max_features': max_features,\n",
        "               'max_depth': max_depth,\n",
        "               'min_samples_split': min_samples_split,\n",
        "               'min_samples_leaf': min_samples_leaf}\n",
        "\n",
        "print(random_grid)"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "{'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], 'max_features': ['auto', 'sqrt'], 'max_depth': [5, 10, 15, 20, 25, 30], 'min_samples_split': [2, 5, 10, 15, 100], 'min_samples_leaf': [1, 2, 5, 10]}\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7ZSw-d0o4xLJ"
      },
      "source": [
        "# Use the random grid to search for best hyperparameters\n",
        "# First create the base model to tune\n",
        "rf = RandomForestRegressor()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qRciJxhm4xLK"
      },
      "source": [
        "# Random search of parameters, using 3 fold cross validation, \n",
        "# search across 100 different combinations\n",
        "rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)"
      ],
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JJDZG8dR4xLK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bba8c2ea-b3a8-4463-acee-d54dbb74461e"
      },
      "source": [
        "#fit the random forest model\n",
        "rf_random.fit(X_train,y_train)"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fitting 5 folds for each of 100 candidates, totalling 500 fits\n",
            "[CV] n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "[CV]  n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.5s remaining:    0.0s\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "[CV]  n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=400, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.5s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.9s\n",
            "[CV] n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   0.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.1s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.3s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.4s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=5, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   0.3s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=10, total=   0.9s\n",
            "[CV] n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1200, min_samples_split=100, min_samples_leaf=10, max_features=auto, max_depth=5, total=   1.4s\n",
            "[CV] n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.3s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   0.6s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.6s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25, total=   1.5s\n",
            "[CV] n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25, total=   1.5s\n",
            "[CV] n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25, total=   1.5s\n",
            "[CV] n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=1100, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=25, total=   1.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.1s\n",
            "[CV] n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   1.0s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.5s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   0.7s\n",
            "[CV] n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   1.0s\n",
            "[CV] n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=800, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.4s\n",
            "[CV] n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.9s\n",
            "[CV] n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=10, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=auto, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=400, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.5s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=5, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=300, min_samples_split=10, min_samples_leaf=5, max_features=auto, max_depth=20, total=   0.4s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=400, min_samples_split=10, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   0.5s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.1s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=auto, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.5s\n",
            "[CV] n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=15, total=   0.5s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   1.3s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   1.3s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.3s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=sqrt, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=5, total=   0.9s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=25, total=   0.2s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=600, min_samples_split=2, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=25, total=   0.9s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.2s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.2s\n",
            "[CV] n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25 \n",
            "[CV]  n_estimators=900, min_samples_split=15, min_samples_leaf=5, max_features=auto, max_depth=25, total=   1.1s\n",
            "[CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.6s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.4s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.6s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=30, total=   1.1s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15 \n",
            "[CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=15, total=   0.4s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   1.5s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=200, min_samples_split=5, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   0.2s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=900, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=30, total=   1.0s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30 \n",
            "[CV]  n_estimators=200, min_samples_split=100, min_samples_leaf=2, max_features=sqrt, max_depth=30, total=   0.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=5, total=   1.3s\n",
            "[CV] n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.0s\n",
            "[CV] n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25 \n",
            "[CV]  n_estimators=700, min_samples_split=2, min_samples_leaf=5, max_features=sqrt, max_depth=25, total=   0.8s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.1s\n",
            "[CV] n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 \n",
            "[CV]  n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   0.1s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10 \n",
            "[CV]  n_estimators=1000, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=10, total=   1.2s\n",
            "[CV] n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=10, min_samples_leaf=1, max_features=auto, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.2s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.3s\n",
            "[CV] n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=200, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.2s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=1200, min_samples_split=10, min_samples_leaf=10, max_features=auto, max_depth=10, total=   1.5s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30 \n",
            "[CV]  n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=auto, max_depth=30, total=   0.7s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   1.4s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=1100, min_samples_split=100, min_samples_leaf=10, max_features=sqrt, max_depth=20, total=   1.3s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.4s\n",
            "[CV] n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10 \n",
            "[CV]  n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_features=auto, max_depth=10, total=   0.4s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.7s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, max_depth=20, total=   0.6s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.9s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n",
            "[CV] n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20 \n",
            "[CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features=sqrt, max_depth=20, total=   0.8s\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=1)]: Done 500 out of 500 | elapsed:  6.5min finished\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomizedSearchCV(cv=5, error_score=nan,\n",
              "                   estimator=RandomForestRegressor(bootstrap=True,\n",
              "                                                   ccp_alpha=0.0,\n",
              "                                                   criterion='mse',\n",
              "                                                   max_depth=None,\n",
              "                                                   max_features='auto',\n",
              "                                                   max_leaf_nodes=None,\n",
              "                                                   max_samples=None,\n",
              "                                                   min_impurity_decrease=0.0,\n",
              "                                                   min_impurity_split=None,\n",
              "                                                   min_samples_leaf=1,\n",
              "                                                   min_samples_split=2,\n",
              "                                                   min_weight_fraction_leaf=0.0,\n",
              "                                                   n_estimators=100,\n",
              "                                                   n_jobs=None, oob_score=Fals...\n",
              "                   iid='deprecated', n_iter=100, n_jobs=1,\n",
              "                   param_distributions={'max_depth': [5, 10, 15, 20, 25, 30],\n",
              "                                        'max_features': ['auto', 'sqrt'],\n",
              "                                        'min_samples_leaf': [1, 2, 5, 10],\n",
              "                                        'min_samples_split': [2, 5, 10, 15,\n",
              "                                                              100],\n",
              "                                        'n_estimators': [100, 200, 300, 400,\n",
              "                                                         500, 600, 700, 800,\n",
              "                                                         900, 1000, 1100,\n",
              "                                                         1200]},\n",
              "                   pre_dispatch='2*n_jobs', random_state=42, refit=True,\n",
              "                   return_train_score=False, scoring='neg_mean_squared_error',\n",
              "                   verbose=2)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ndP9KkRN4xLM",
        "outputId": "7859d918-970f-402e-8460-c31227b848bd"
      },
      "source": [
        "#displaying the best parameters\n",
        "rf_random.best_params_"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'n_estimators': 300,\n",
              " 'min_samples_split': 5,\n",
              " 'min_samples_leaf': 2,\n",
              " 'max_features': 'auto',\n",
              " 'max_depth': 10}"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WLM_k5fo4xLN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "98916a52-2b69-44d2-f0ae-34cf4677e465"
      },
      "source": [
        "rf_random.best_score_"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "-3.693969317699947"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0FMntHcE4xLN"
      },
      "source": [
        "# Final Predictions"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jKXNJBmV4xLO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "704778fe-be89-49d6-b02d-05795b5051f6"
      },
      "source": [
        "#predicting against test data\n",
        "y_pred=rf_random.predict(X_test)\n",
        "#print the erros\n",
        "print('MAE:', metrics.mean_absolute_error(y_test, y_pred))\n",
        "print('MSE:', metrics.mean_squared_error(y_test, y_pred))\n",
        "print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))\n",
        "R2 = metrics.r2_score(y_test,y_pred)\n",
        "print('R2:',R2)"
      ],
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: 0.7640338013156972\n",
            "MSE: 2.6122079318899827\n",
            "RMSE: 1.6162326354488648\n",
            "R2: 0.9126528797365616\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g52-Tt2u4xLO"
      },
      "source": [
        "# Save the model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "k2hLj5J34xLP"
      },
      "source": [
        "import pickle\n",
        "# open a file, where you ant to store the data\n",
        "file = open('car_price_model.pkl', 'wb')\n",
        "\n",
        "# dump information to that file\n",
        "pickle.dump(rf_random, file)"
      ],
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WRNOcJEU4xLP"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}