/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */
package de.jdreisvogt.sandbox.neuralnet.layer;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;
import de.jdreisvogt.sandbox.neuralnet.neuron.TooManyInputsException;

public abstract class Layer
{
    protected NeuralNet parent;
    private final Function function;
    protected Neuron[] neurons;
    private double bias;
    private final int id;
    private Layer nextLayer;
    private double[] lastInput;
    protected double[] lastOutput;

    /**
     * Repräsentiert eine Aktivierungsfunktion
     */
    public interface Function
    {
        /**
         * Die Funktion selbst
         * @param x Eingabe (x-Wert)
         * @return Ausgabe (f(x)-Wert)
         */
        double compute(double x);

        /**
         * Die Ableitung der Funktion
         * @param x Eingabe (x-Wert)
         * @return Ausgabe (f'(x)-Wert)
         */
        double computeDerivative(double x);

        /**
         * Gibt einen eindeutigen Namen der Funktion zurück
         * @return Name
         */
        String getAlgorithmName();
    }

    /**
     * Konstruktor
     * @param parent Das KNN, zu welchem der Layer gehört
     * @param id Eine eindeutige Identifikationsnummer. Muss dem Index im KNN entsprechen.
     * @param size Anzahl der Neuronen
     * @param function Aktivierungsfunktion
     * @param nextLayer Verweis auf den nachfolgenden Layer
     */
    public Layer(NeuralNet parent, int id, int size, Function function, Layer nextLayer)
    {
        this.parent = parent;
        this.id = id;
        this.function = function;
        this.neurons = new Neuron[size];
        this.nextLayer = nextLayer;
    }

    /**
     * Generiert die Neuronen für den Layer
     * Gewichte werden zufällig in einem Bereich gewählt
     * @param max_input_size Die maximale Größe der Eingaben, welche die Neuronen verarbeiten können (Anzahl von deren Gewichten)
     * @param min_weight_value Der kleinste Wert des Bereiches, in dem die Gewichte initialisiert werden
     * @param max_weight_value Der größte Wert des Bereiches, in dem die Gewichte initialisiert werden
     */
    public void createNeurons(int max_input_size, double min_weight_value, double max_weight_value)
    {
        for (int i = 0; i < this.neurons.length; i++)
        {
            this.neurons[i] = this.createNeuron(max_input_size, i, min_weight_value, max_weight_value);
        }
    }

    protected abstract Neuron createNeuron(int max_input_size, int id, double min_weight_value, double max_weight_value);

    /**
     * Berechnet die Ausgabe für die übergebene Eingabe
     * @param x Eingabe (Ausgabe der vorherigen Schicht)
     * @return Ausgabe des Layers
     * @throws TooManyInputsException falls die maximale Größe der Eingabe überschritten wurde
     */
    public double[] compute(double[] x) throws TooManyInputsException
    {
        this.lastInput = x;
        double[] result = new double[this.neurons.length];
        for (int i = 0; i < this.neurons.length; i++)
        {
            result[i] = this.neurons[i].compute(x);
        }
        this.lastOutput = result;
        return result;
    }

    /**
     * Ruft die Methode für den Backpropagation-Algorithmus auf den Neuronen auf
     */
    public void train()
    {
        for (Neuron neuron : this.neurons)
        {
            neuron.changeWeights();
        }
    }

    //getter and setter functions:

    public double[] getLastInput()
    {
        return lastInput;
    }

    public double[] getLastOutput()
    {
        return lastOutput;
    }

    public int getSize()
    {
        return this.neurons.length;
    }

    public void setNextLayer(Layer nextLayer)
    {
        this.nextLayer = nextLayer;
    }

    public Layer getNextLayer()
    {
        return this.nextLayer;
    }

    public Neuron getNeuron(int id)
    {
        if(neurons.length <= id) return null;
        return this.neurons[id];
    }

    public NeuralNet getNet()
    {
        return this.parent;
    }

    public double getBias()
    {
        return this.bias;
    }

    public int getLayerId()
    {
        return this.id;
    }

    public Function getFunction()
    {
        return this.function;
    }

    public void setBias(double bias)
    {
        this.bias = bias;
    }

    public double getTolerance()
    {
        return this.parent.getTolerance();
    }
}