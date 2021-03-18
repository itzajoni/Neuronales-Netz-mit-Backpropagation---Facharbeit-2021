/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.net;

import de.jdreisvogt.sandbox.neuralnet.layer.Layer;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;
import de.jdreisvogt.sandbox.neuralnet.test.MnistLoader;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.data.Range;
import org.jfree.data.category.DefaultCategoryDataset;


import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;

public class NeuralNetAnalyzer
{
    /**
     * Gibt Spezifikationen an, welche Graphen gedruckt werden sollen
     */
    public static class ViewSpecs
    {
        public boolean showError = false;
        public boolean showUnclassified = true;
        public boolean showAbsolute = false;
        public boolean showAbsolutePerLabel = false;
        public boolean showTrainingCorrect = false;
        public boolean showTrainingFalse = false;
        public boolean showTrainingUnclassified = false;

    }

    /**
     * Erstellt ein Diagram, auf dem Größen aus NeuralNetTrainingResults dargestellt werden
     * @param output_path Dateipfad, an dem das Diagram als .png-Datei gespeichert werden soll
     * @param results HashMap, die jeden Datensatz einem Namen zuordnet
     * @param viewSpecs Spezifikationen des Diagramms
     * @param y_limit Der Abschnitt, der auf der Y-Achse dargestellt werden soll.
     */
    public static void createComparisonDiagram (String output_path, HashMap<String, NeuralNet.NeuralNetTrainingResult> results, ViewSpecs viewSpecs, double y_limit)
    {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.correct; }}, "Richtige Klassifizierungen", false, dataset);
        if (viewSpecs.showError) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.wrong; }}, "Falsche Klassifizierungen", false, dataset);
        if (viewSpecs.showUnclassified) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.unclassified; }}, "Nicht klassifiziert", false, dataset);
        if (viewSpecs.showTrainingCorrect) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.correct; }}, "Richtige Klassifizierungen Training", true, dataset);
        if (viewSpecs.showTrainingFalse) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.wrong; }}, "Falsche Klassifizierungen Training", true, dataset);
        if (viewSpecs.showTrainingUnclassified) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {return result.unclassified; }}, "Nicht klassifiziert Training", true, dataset);
        if (viewSpecs.showAbsolute) generateCollection(results, new PropertyAccesser() {@Override public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result) {
            double error = 0;
            for (int i = 0; i < result.absolute_output_error.length; i++)
            {
                error += result.absolute_output_error[i];
            }
            return error;
        }}, "Richtige Klassifizierungen", false, dataset);
        if (viewSpecs.showAbsolutePerLabel) for (int i = 0; i < 10; i++)
        {
            int finalI = i;
            generateCollection(results, new PropertyAccesser()
            {
                @Override
                public double returnValue (NeuralNet.NeuralNetTrainingResult.Result result)
                {
                    return ((result.correct_per_label[finalI] + 0.0) / result.total_per_label[finalI]) + 0.0;
                }
            }, "Richtig: " + i, false, dataset);
        }

        JFreeChart chart = ChartFactory.createLineChart("NeuralNetTrainingResult:", "", "",dataset);
        chart.setBackgroundPaint(Color.white);
        CategoryPlot plot = (CategoryPlot) chart.getPlot();
        plot.getRangeAxis().setRange(new Range(0, y_limit));
        plot.setBackgroundPaint(Color.white);
        for (int i = 0; i < 100; i++)
        {
            plot.getRenderer().setSeriesStroke(i, new BasicStroke(5));
        }
        try
        {
            ChartUtils.saveChartAsPNG(new File(output_path), chart, 1280, 720);
        } catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    private static void generateCollection (HashMap<String, NeuralNet.NeuralNetTrainingResult> results, PropertyAccesser accesser, String name, boolean training, DefaultCategoryDataset dataset)
    {
        for (String s : results.keySet())
        {
            NeuralNet.NeuralNetTrainingResult trainingResult = results.get(s);
            for (int i = 0; i < trainingResult.getLength(); i++)
            {
                dataset.addValue(accesser.returnValue(training ? trainingResult.trainingResults[i] : trainingResult.testResults[i]), name + "(" + s + ")", i + "");
            }
        }
    }

    private static abstract class PropertyAccesser
    {
        public abstract double returnValue (NeuralNet.NeuralNetTrainingResult.Result result);
    }

    /**
     * Vergleicht zwei KNNs auf ihre Gewichte; ausschließlich für Abschnitt 3.2 benötigt
     * Achtung: Fehler bei der Berechnung der Genauigkeit der Netze; unklar ob vollständig behoben
     * Achtung: Die Netze müssen die gleiche Struktur haben
     * @param net1 Das erste Netz
     * @param net2 Das zweite Netz
     */
    public static void compareWeights (NeuralNet net1, NeuralNet net2)
    {
        int total_weights = 0;
        double[] average1 = new double[net1.getHiddenLayers().size() + 1];
        double[] average2 = new double[net1.getHiddenLayers().size() + 1];
        double[] highest_weight = new double[net1.getHiddenLayers().size() + 1];
        double[] lowest_weight = new double[net1.getHiddenLayers().size() + 1];
        double[] biggest_differenz = new double[net1.getHiddenLayers().size() + 1];
        int correct1 = 0;
        int correct2 = 0;
        int different_results = 0;

        for (int i = 0; i < net1.getHiddenLayers().size() + 1; i++)
        {
            highest_weight[i] = Integer.MIN_VALUE;
            lowest_weight[i] = Integer.MAX_VALUE;
            average1[i] = 0;
            average2[i] = 0;
            biggest_differenz[i] = -1;
            Layer layer1 = i != net1.getHiddenLayers().size() ? net1.getHiddenLayers().get(i) : net1.getOutputLayer();
            Layer layer2 = i != net1.getHiddenLayers().size() ? net2.getHiddenLayers().get(i) : net2.getOutputLayer();
            for (int j = 0; j < layer1.getSize(); j++)
            {
                Neuron neuron1 = layer1.getNeuron(j);
                Neuron neuron2 = layer2.getNeuron(j);
                for (int k = 0; k < (i == 0 ? net1.getInputSize() : net1.getHiddenLayers().get(i - 1).getSize()); k++)
                {
                    double weight1 = neuron1.getWeight(k);
                    double weight2 = neuron2.getWeight(k);
                    if (weight1 > highest_weight[i]) highest_weight[i] = weight1;
                    if (weight2 > highest_weight[i]) highest_weight[i] = weight2;
                    if (weight1 < lowest_weight[i]) lowest_weight[i] = weight1;
                    if (weight2 < lowest_weight[i]) lowest_weight[i] = weight2;
                    if (Math.abs(weight1 - weight2) > biggest_differenz[i]) biggest_differenz[i] = Math.abs(weight1 - weight2);
                    average1[i] += weight1;
                    average2[i] += weight2;
                    total_weights++;
                }
            }
            average1[i] /= layer1.getSize();
            average2[i] /= layer1.getSize();
        }

        MnistLoader loader = new MnistLoader();
        loader.load();
        MnistLoader.DataSet[] test = loader.getAll();
        for (MnistLoader.DataSet dataSet : test)
        {
            int res1 = analyzeResult(net1.compute(dataSet.x));
            int res2 = analyzeResult(net2.compute(dataSet.x));
            if (!(res1 <= 0 || dataSet.y[res1] != 1)) correct1++;
            if (!(res2 <= 0 || dataSet.y[res2] != 1)) correct2++;
            if (res1 != res2) different_results++;
        }

        System.out.println("Vergleich der KNNs abgeschlossen:");
        for (int i = 0; i < 2; i++)
        {
            System.out.println("Genauigkeit Netz " + i + ": " + (i == 1 ? correct1 : correct2) + "/42000");
            System.out.println("Durchschnittliche Gewichte Netz " + i + ": " + arrayToString(i == 1 ? average1 : average2));

        }
        System.out.println("Größte Gewichte: " + arrayToString(highest_weight));
        System.out.println("Kleinste Gewichte: " + arrayToString(lowest_weight));
        System.out.println("Größte Unterschiede: " + arrayToString(biggest_differenz));
        System.out.println("Gewichte pro Netz insgesamt: " + total_weights);
        System.out.println("Unterschiedliche Ergebnisse: " + different_results);
    }

    private static String arrayToString (double[] stats)
    {
        StringBuilder result = new StringBuilder("[");
        for (double stat : stats)
        {
            result.append(stat).append(", ");
        }
        return result + "]";
    }

    private static int analyzeResult (double[] nn_output)
    {
        int highest_index = 0;
        double highest_value = Integer.MIN_VALUE;
        double sec_highest_value = Integer.MIN_VALUE;

        for (int k = 0; k < nn_output.length; k++)
        {
            if (nn_output[k] > highest_value)
            {
                highest_index = k;
                sec_highest_value = highest_value;
                highest_value = nn_output[k];
            } else if (sec_highest_value < nn_output[k]) sec_highest_value = nn_output[k];
        }

        if (highest_value < sec_highest_value + 0.2) return -1;
        return highest_index;
    }
}
