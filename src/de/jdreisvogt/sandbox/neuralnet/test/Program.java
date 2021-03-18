/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.net.NeuralNetAnalyzer;
import de.jdreisvogt.sandbox.neuralnet.net.NeuralNetSaver;
import org.json.simple.parser.ParseException;

import java.util.HashMap;

/**
 * Wurde während den Untersuchungen genutzt, um die in der Facharbeit und im Anhang verwendeten Daten zu berechnen.
 * Achtung: Es sind nicht mehr alle verwendeten Programme vorhanden.
 */
public class Program
{
    private static MnistLoader loader;

    public static void main (String[] args)
    {
        runMultiple();
    }

    public static void getDifferentLabelTrainingDiagram ()
    {
        loader = new MnistLoader();
        loader.load();
        System.out.println("data loaded");
        HashMap<String, NeuralNet.NeuralNetTrainingResult> nets = new HashMap<>();

        NeuralNet.NeuralNetTrainingResult[] results = new NeuralNet.NeuralNetTrainingResult[30];
        for (int i = 0; i < results.length; i++)
        {
            //results[i] = execute(new int[]{200, 10}, "referenz(200+10) lr0.01 t0.02 100r 37000+5000", 0.02, 0.01, 37000, 5000, 50);
        }
        System.out.println("Training done");
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                int sum = 0;
                for (int k = 0; k < 30; k++)
                {
                    sum += results[k].testResults[i].correct_per_label[j];
                }
                results[0].testResults[i].correct_per_label[j] = sum / 30;
            }
        }
        nets.put("200+10", results[0]);
        NeuralNetAnalyzer.ViewSpecs specs = new NeuralNetAnalyzer.ViewSpecs();
        specs.showUnclassified = false;
        specs.showAbsolutePerLabel = true;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\experimental.png", nets, specs, 1);
    }

    public static void createDifferentLearningRatesDiagram ()
    {
        HashMap<String, NeuralNet.NeuralNetTrainingResult> results = new HashMap<>();
        results.put("lr 0.00001", create(new int[]{200}, 0.02, 0.00001, 37000, 5000, 100));
        results.put("lr 0.01", create(new int[]{200}, 0.02, 0.01, 37000, 5000, 100));
        results.put("lr 1", create(new int[]{200}, 0.02, 1, 37000, 5000, 100));
        NeuralNetAnalyzer.ViewSpecs specs = new NeuralNetAnalyzer.ViewSpecs();
        specs.showUnclassified = false;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentLearningRatesCorrect.png", results, specs, 5000);
        specs.showUnclassified = true;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentLearningRatesUnclassified.png", results, specs, 5000);
        specs.showUnclassified = false;
        specs.showError = true;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentLearningRatesWrong.png", results, specs, 5000);
    }

    public static void createDataDiagram ()
    {
        HashMap<String, NeuralNet.NeuralNetTrainingResult> results = new HashMap<>();
        results.put("41000", create(new int[]{200}, 0.02, 0.01, 41000, 1000, 100));
        results.put("10000", create(new int[]{200}, 0.02, 0.01, 10000, 1000, 100));
        results.put("1000", create(new int[]{200}, 0.02, 0.01, 1000, 1000, 100));
        NeuralNetAnalyzer.ViewSpecs specs = new NeuralNetAnalyzer.ViewSpecs();
        specs.showUnclassified = false;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentTrainDataCorrect.png", results, specs, 1000);
        specs.showUnclassified = true;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentTrainDataUnclassified.png", results, specs, 1000);
        specs.showUnclassified = false;
        specs.showError = true;
        NeuralNetAnalyzer.createComparisonDiagram("C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\differentTrainDataWrong.png", results, specs, 1000);
    }

    public static void runMultiple ()
    {
        //Aus Performancegründen werden mehrere Threads erstellt
        loader = new MnistLoader();
        loader.load();
        //new Thread(Program::createDifferentLearningRatesDiagram).start();
        new Thread(Program::createDataDiagram).start();
        /*new Thread(() ->{

        }).start();
        new Thread(() ->{

            System.out.println("Thread fertig");
        }).start();
        new Thread(() ->{

            System.out.println("Thread fertig");
        }).start();
        System.out.println("Alle Threads gestartet");

        /*
        altes Program:
        new Thread(() -> {
            execute(new int[]{200, 10}, "referenz(200+10) lr0.01 t0.02 100r 37000+5000", 0.02, 0.01, 37000, 5000, 100);
            execute(new int[]{200, 10}, "(200+10) lr0.001 t0.02 100r 37000+5000", 0.02, 0.001, 37000, 5000, 100);
            System.out.println("Thread fertig");
        }).start();
        new Thread(() -> {
            execute(new int[]{10}, "(10) lr0.01 t0.02 100r 37000+5000", 0.02, 0.01, 37000, 5000, 100);
            execute(new int[]{200, 10}, "(200+10) lr0.1 t0.02 100r 37000+5000", 0.02, 0.1, 37000, 5000, 100);
            execute(new int[]{200, 10}, "referenz(200+10) lr0.01 t0.02 100r 5000+37000", 0.02, 0.01, 5000, 37000, 10);
        }).start();
        new Thread(() ->{
            execute(new int[]{200,200,10},"(200+200+10) lr0.01 t0.02 100r 37000+5000",0.02,0.01,37000,5000,100);
            execute(new int[]{200,10},"(200+10) lr0.01 t0.2 100r 37000+5000",0.2,0.01,37000,5000,100);
            System.out.println("Thread fertig");
        }).start();
        new Thread(() ->{
            execute(new int[]{50,50,50,50},"(50+50+50+50) lr0.01 t0.02 100r 37000+5000",0.02,0.01,37000,5000,100);
            execute(new int[]{200,10},"(200+10) lr0.01 t0 100r 37000+5000",0,0.01,37000,5000,100);
            System.out.println("Thread fertig");
        }).start();
        new Thread(() ->{
            execute(new int[]{200},"referenz(200) lr0.01 t0.02 100r 37000+5000",0.02,0.01,37000,5000,100);
            execute(new int[]{200,10},"referenz(200+10) lr0.01 t0.02 1000r 37000+5000",0.02,0.01,37000,5000,1000);
            System.out.println("Thread fertig");
        }).start();
        new Thread(() ->{
            execute(new int[]{200,10},"referenz(200+10) lr1 t0.02 100r 37000+5000",0.02,1,37000,5000,100);
            execute(new int[]{200,10},"referenz(200+10) lr0.01 t0.02 100r 41000+1000",0.02,0.01,41000,1000,100);
            System.out.println("Thread fertig");
        }).start();
         */
    }


    public static NeuralNet.NeuralNetTrainingResult create(int[] layers, double tolerance, double learning_rate, int train_size, int test_size, int iterations)
    {

        NeuralNet net = new NeuralNet(784);
        net.setTolerance(tolerance);
        net.setLearningRate(learning_rate);
        net.setStandardBias(0);
        for (int layer : layers)
        {
            net.addLayer(layer, Functions.SIGMOID.function);
        }
        net.generateOutputLayer(10, Functions.SIGMOID.function);
        System.out.println("net ready");

        MnistLoader.DataSet[] training = loader.getPart(0, train_size - 1);
        MnistLoader.DataSet[] test = loader.getPart(42000 - test_size , 41999);

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

        //print = true;
        System.out.println("data ready");
        System.out.println("start training");
        NeuralNet.NeuralNetTrainingResult test_result = net.train(training_x, training_y, test_x, test_y, iterations);

        System.out.println("result:");
        System.out.print(test_result);
        return test_result;

        /*if (name != null)
        {
            NeuralNetSaver.SaveNeuralNet(net, "C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\" + name + ".nnet", false);
            NeuralNetSaver.SaveTrainingResult(test_result, "C:\\Users\\jonat\\Desktop\\Facharbeit\\NNs\\" + name + ".log");
        }

        return net;
        /*for (int i = 0; i < 20; i++)
        {
            GUI gui = new GUI(test[i].correctValue);
            double[][] pixel = new double[28][28];
            for (int j = 0; j < 28; j++)
            {
                double[] row = new double[28];
                System.arraycopy(test_x[i], (j * 28), row, 0, 28);
                pixel[j] = row;
            }
            gui.setPixel(pixel);
            gui.setResults(net.compute(test_x[i]));
        }*/
    }
}
