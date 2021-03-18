/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test.demo;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.net.NeuralNetAnalyzer;
import de.jdreisvogt.sandbox.neuralnet.net.NeuralNetSaver;
import de.jdreisvogt.sandbox.neuralnet.test.Functions;
import de.jdreisvogt.sandbox.neuralnet.test.MnistLoader;

import java.util.HashMap;

/**
 * Demo-Programm #3
 * Erstellt ein Diagramm über den Lernerfolg zweier KNN
 * Zur Ausführung muss die unter /code/Datenbank-Erstellen/datenbank.sql gespeicherte Datenbank erstellt und unter
 * folgenden Parametern verfügbar sein:
 * Name: 'Facharbeit', Port: 3306, SQL-Dialekt: MYSQL, Benutzername: 'root', Passwort: ''
 */
public class Demo3
{
    /**
     * Die in createGraphics(...) erstellte Grafik wird in einer Datei gespeichert
     * Es ist dazu erforderlich, dass das Verzeichnis existent ist.
     * @param args ignored
     */
    public static void main (String[] args)
    {
        createGraphics("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\diagram");
    }

    /**
     * Erstellt eine Grafik über den Lernerfolg zweier Netze (200+10+10 und 50+10+10Neuronen; Referenzbedingungen)
     * @param path Pfad, unter dem die Grafik gespeichert wird
     */
    public static void createGraphics(String path)
    {
        HashMap<String, NeuralNet.NeuralNetTrainingResult> results = new HashMap<>();
        System.out.println("Netz 1 (200+10+10) wird erstellt: ");
        results.put("200+10", create(new int[]{200, 10}, null, 0.02, 0.01, 37000, 5000, 20));
        System.out.println("");
        System.out.println("Netz 2 (50+10+10) wird erstellt: ");
        results.put("50+10", create(new int[]{50, 10}, null, 0.02, 0.01, 37000, 5000, 20));
        NeuralNetAnalyzer.createComparisonDiagram(path + ".png", results, new NeuralNetAnalyzer.ViewSpecs(), 5000);
    }

    /**
     * Erstellt ein KNN, trainiert es mit den angegebenen Parametern und speichert es evtl. ab
     * @param layers Die Anzahl und die Größe der Layer
     * @param path Der Pfad, unter dem das KNN gespeichert wird. Das Verzeichnis muss existieren. Wird nicht gespeichert wenn null
     * @param tolerance Die Toleranz während des Lernprozesses
     * @param learning_rate Die Lernrate
     * @param train_size Die Anzahl der Daten, mit denen Trainiert wird
     * @param test_size Die Anzahl der Daten, mit denen getestet wird
     * @param iterations Die Anzahl an Trainingsdurchläufen.
     * @return Die Analysedaten des Trainings
     */
    public static NeuralNet.NeuralNetTrainingResult create(int[] layers, String path, double tolerance, double learning_rate, int train_size, int test_size, int iterations)
    {
        MnistLoader source = new MnistLoader();
        try
        {
            source.load();
        } catch (Exception e)
        {
            System.out.println("Das Programm konnte keine Datenbank mit den richtigen Spezifikationen entdecken.");
            System.out.println("Benötigte Spezifikationen: Name: 'Facharbeit', Port: 3306, SQL-Dialekt: MYSQL, Benutzername: 'root', Passwort: ''");
            System.out.println("Die DB muss eine Tabelle 'mnistimgtbl' enthalten, welche als sql Datei in diesem Anhang gespeichert ist und importiert werden kann");
            return new NeuralNet.NeuralNetTrainingResult();
        }

        System.out.println("Daten wurden geladen");
        NeuralNet net = new NeuralNet(784);
        net.setTolerance(tolerance);
        net.setLearningRate(learning_rate);
        net.setStandardBias(0);
        for (int layer : layers)
        {
            net.addLayer(layer, Functions.SIGMOID.function);
        }
        net.generateOutputLayer(10, Functions.SIGMOID.function);
        System.out.println("Netz wurde erstellt");

        MnistLoader.DataSet[] training = source.getPart(0, train_size - 1);
        MnistLoader.DataSet[] test = source.getPart(42000 - test_size, 41999);

        double[][] training_x = new double[train_size - 1][784];
        double[][] training_y = new double[train_size - 1][10];
        double[][] test_x = new double[test_size - 1][784];
        double[][] test_y = new double[test_size - 1][10];

        for (int i = 0; i < training.length; i++)
        {
            training_x[i] = training[i].x;
            training_y[i] = training[i].y;
        }

        for (int i = 0; i < test.length; i++)
        {

            test_x[i] = test[i].x;
            test_y[i] = test[i].y;
        }

        System.out.println("Daten wurden vorbereitet");
        System.out.println("Training wird gestartet");
        NeuralNet.NeuralNetTrainingResult test_result = net.train(training_x, training_y, test_x, test_y, iterations);

        if (path != null)
        {
            NeuralNetSaver.SaveNeuralNet(net, path + ".nnet", false);
            NeuralNetSaver.SaveTrainingResult(test_result, path + ".log");
        }

        return test_result;
    }
}
