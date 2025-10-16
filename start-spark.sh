#!/bin/zsh
# start-spark.sh
# This script runs Spark shell with Java 17 on macOS without affecting Hadoop Java version

# Set Java 17 for this session
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

# Optional: print java version to confirm
echo "Using Java version:"
java -version

# Run Spark shell
spark-shell --master "local[*]" --conf spark.ui.port=4041
