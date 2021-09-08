import pyspark.sql.functions as f
from pyspark.sql import SparkSession, WindowSpec, Window, DataFrame, Column

from minsait.ttaa.datio.common.Constants import *
from minsait.ttaa.datio.common.naming.PlayerInput import *
from minsait.ttaa.datio.common.naming.PlayerOutput import *
from minsait.ttaa.datio.utils.Writer import Writer


class Transformer(Writer):
    def __init__(self, spark: SparkSession):
        self.spark: SparkSession = spark
        df: DataFrame = self.read_input()
        df.printSchema()
        df = self.column_selection(df)
        df = self.example_window_function(df)
        df = self.get_potential_vs_overall(df)
        df = self.evaluate_players(df)
        self.test_evaluate_players_player_cat(df)

        # for show 100 records after your transformations and show the DataFrame schema
        df.show(n=100, truncate=False)
        df.printSchema()

        # Uncomment when you want write your final output
        self.write(df)

    def read_input(self) -> DataFrame:
        """
        :return: a DataFrame readed from csv file
        """
        return self.spark.read \
            .option(INFER_SCHEMA, True) \
            .option(HEADER, True) \
            .csv(INPUT_PATH)

    def evaluate_players(self, df: DataFrame) -> DataFrame:
        """
        :param df: is a DataFrame with players information
        :return: a DataFrame with filter transformation applied
        column team_position != null && column short_name != null && column overall != null
        """
        df = df.filter(
            (player_cat.column().isin("A","B")) |
            ((player_cat.column() == "C") & (potential_vs_overall.column() > 1.15)) |
            ((player_cat.column() == "D") & (potential_vs_overall.column() > 1.25))
        )
        return df


    def test_data(df1: DataFrame, df2: DataFrame):
            """
            Function for comparing two DataFrame data. If data is equal returns True.
            :param df1: test DataFrame :param df2: test DataFrame
            :return: Boolean
            """

            data1 = df1.collect()
            data2 = df2.collect()
            return set(data1) == set(data2)

    def test_evaluate_players_player_cat(self, df: DataFrame):
        evaluated_df=df.select(player_cat.column()).dropDuplicates()
        expected_Df=self.spark.createDataFrame(
            data=[['A'],['B'],
                  ['C'],['D']],
            schema=['player_cat'])
        self.asserTrue(self.test_data(evaluated_df, expected_Df))



    def test_evaluate_players_player_cat_C(self, df: DataFrame):
        test_df= df.filter(player_cat.column()=="C").select(potential_vs_overall.column())
        wrong_df_filter=test_df.filter(potential_vs_overall.column()<=1.15)
        self.assert wrong_df_filter is None

    def column_selection(self, df: DataFrame) -> DataFrame:
        """
        :param df: is a DataFrame with players information
        :return: a DataFrame with just 5 columns...
        """
        df = df.select(
            short_name.column(),
            long_name.column(),
            age.column(),
            height_cm.column(),
            weight_kg.column(),
            nationality.column(),
            club_name.column(),
            overall.column(),
            potential.column(),
            team_position.column(),
        )
        return df

    def example_window_function(self, df: DataFrame) -> DataFrame:
        """
        :param df: is a DataFrame with players information (must have team_position and height_cm columns)
        :return: add to the DataFrame the column "cat_height_by_position"
             by each position value
             cat A for if is in 20 players tallest
             cat B for if is in 50 players tallest
             cat C for the rest
        """
        w: WindowSpec = Window \
            .partitionBy(nationality.column(),\
                        team_position.column()) \
            .orderBy(overall.column().asc())
        rank: Column = f.rank().over(w)

        rule: Column = f.when(rank <= 3, "A") \
            .when(rank <= 5, "B") \
            .when(rank <= 10, "C") \
            .otherwise("D")

        df = df.withColumn(player_cat.name, rule)
        return df

    def get_potential_vs_overall(self, df: DataFrame) -> DataFrame:
        """
        :param df: is a DataFrame with players information
        :return: a DataFrame add potential_vs_overall column
        """
        df = df.withColumn(
            potential_vs_overall.name,
            potential.column()/overall.column()
        )
        return df