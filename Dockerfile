FROM openjdk:8-jdk-alpine
COPY . .
WORKDIR /src
RUN javac main/Main.java
CMD ["java", "main/Main"]