/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.net;

import de.jdreisvogt.sandbox.neuralnet.layer.HiddenLayer;
import de.jdreisvogt.sandbox.neuralnet.layer.Layer;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;
import de.jdreisvogt.sandbox.neuralnet.test.Functions;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.awt.Toolkit;

/**
 * Dient zur Persistenten Speicherung der KNNs sowie der TrainingResults
 */
public class NeuralNetSaver
{
    private static final JSONParser parser = new JSONParser();

    /**
     * Wandelt das KNN in einen String um
     * @param net KNN
     * @return String-Value des KNN (JSON-Format)
     */
    public static String serializeNet(NeuralNet net)
    {
        JSONObject result = new JSONObject();
        if (net.getHiddenLayers().isEmpty()) result.put("input-size", net.getOutputLayer().getNeuron(0).getData().weights.length);
        else result.put("input-size", net.getHiddenLayers().get(0).getNeuron(0).getData().weights.length);
        result.put("tolerance", net.getTolerance());
        result.put("learning-rate", net.getLearningRate());
        JSONArray layers = new JSONArray();
        ArrayList<HiddenLayer> hidden = net.getHiddenLayers();
        for (HiddenLayer hiddenLayer : hidden)
        {
            layers.add(serializeLayer(hiddenLayer));
        }
        result.put("hidden", layers);
        result.put("output", serializeLayer(net.getOutputLayer()));

        return result.toJSONString();
    }

    /**
     * Erstellt ein KNN aus einem mit serializeNet(...) erstellten String
     * @param s String-Value des KNN
     * @return KNN
     * @throws ParseException Wenn String ungültig ist
     */
    public static NeuralNet Load(String s) throws ParseException
    {
        JSONObject data = (JSONObject) parser.parse(s);
        NeuralNet net = new NeuralNet(((Long) data.get("input-size")).intValue());
        net.setLearningRate((double) data.get("learning-rate"));
        net.setTolerance((double) data.get("tolerance"));
        JSONArray layers_data = (JSONArray) data.get("hidden");
        for (Object layer_data_ : layers_data)
        {
            JSONObject layer_data = (JSONObject) layer_data_;
            Functions function = Functions.getByAlgorithmName((String) layer_data.get("function"));
            if (function == null)
            {
                System.out.println("function \"" + layer_data.get("function") + "\" is not in function list. Replaced with Sigmoid function.");
                function = Functions.SIGMOID;
            }
            net.addLayer(getMatrixFromJSON((JSONArray) layer_data.get("neurons")), function.function, (double) layer_data.get("bias"));
        }
        JSONObject layer_data = (JSONObject) data.get("output");
        Functions function = Functions.getByAlgorithmName((String) layer_data.get("function"));
        if (function == null)
        {
            System.out.println("function \"" + layer_data.get("function") + "\" is not in function list. Replaced with Sigmoid function.");
            function = Functions.SIGMOID;
        }
        net.generateOutputLayer(getMatrixFromJSON((JSONArray) layer_data.get("neurons")), function.function, (double) layer_data.get("bias"));

        return net;
    }

    private static double[][] getMatrixFromJSON(JSONArray array)
    {
        double[][] weights = new double[array.size()][((JSONArray) array.get(0)).size()];
        for (int i = 0; i < weights.length; i++)
        {
            JSONArray neuron = (JSONArray) array.get(i);
            for (int j = 0; j < weights[0].length; j++)
            {
                weights[i][j] = (double) neuron.get(j);
            }
        }
        return weights;
    }

    private static JSONObject serializeLayer(Layer layer)
    {
        JSONObject layer_data = new JSONObject();
        JSONArray neurons = new JSONArray();
        for (int j = 0; j < layer.getSize(); j++)
        {
            JSONArray weights = new JSONArray();
            Neuron.NeuronData neuron = layer.getNeuron(j).getData();
            for (int k = 0; k < neuron.weights.length; k++)
            {
                weights.add(neuron.weights[k]);
            }
            neurons.add(weights);
        }
        layer_data.put("neurons", neurons);
        layer_data.put("bias", layer.getBias());
        layer_data.put("function", layer.getFunction().getAlgorithmName());
        return layer_data;
    }

    private static void writeToFile(String path, String content, boolean copy_to_clipboard)
    {
        try
        {
            File target = new File(path);
            target.createNewFile();
            FileWriter writer = new FileWriter(path);
            writer.write(content);
            writer.close();

            if (copy_to_clipboard)
            {
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                Transferable string = new StringSelection(content);
                clipboard.setContents(string, null);
            }
        } catch (IOException e)
        {
            System.out.println("cannot write file: " + e);
            e.printStackTrace();
        }
    }

    private static String readFromFile(String path)
    {
        try
        {
            StringBuilder result = new StringBuilder();
            File file = new File(path);
            if (!file.exists()) throw new IOException("File does not exist");
            Scanner reader = new Scanner(file);
            while (reader.hasNextLine())
            {
                result.append(reader.nextLine()).append('\n');
            }
            return result.toString();
        } catch (IOException e)
        {
            System.out.println("cannot read file: " + e.getMessage());
            e.printStackTrace();
            return "";
        }
    }

    /**
     * Serialisiert ein KNN und speichert es in einer Datei ab
     * @param net KNN
     * @param path Pfad zur Datei
     * @param copy_to_clipboard Ob der Inhalt der Datei in die Zwieschenablage kopiert werden soll
     */
    public static void SaveNeuralNet(NeuralNet net, String path, boolean copy_to_clipboard)
    {
        writeToFile(path, serializeNet(net), copy_to_clipboard);
    }

    /**
     * Speichert ein TrainingResult in einer Datei ab
     * @param result TrainingResult
     * @param path Pfad zur Datei
     */
    public static void SaveTrainingResult(NeuralNet.NeuralNetTrainingResult result, String path)
    {
        writeToFile(path, result.toString(), true);
    }

    /**
     * Lädt ein KNN aus einer Datei
     * @param path Pfad zur Datei
     * @return KNN
     * @throws ParseException falls Datei invalide ist.
     */
    public static NeuralNet LoadNet(String path) throws ParseException
    {
        return Load(readFromFile(path));
    }
}
