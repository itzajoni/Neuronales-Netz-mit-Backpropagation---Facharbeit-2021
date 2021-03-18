/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.neuron;

import de.jdreisvogt.sandbox.neuralnet.layer.Layer;

public abstract class Neuron
{
    protected double[] weights;       //the weights for the input vector
    protected Layer layer;            //the parent neuron layer (input-layer, hidden-layer or output-layer)
    protected int neuronId;           //the id of the neuron, used for debugging
    protected double lastComputedErrorSignal;
    protected double lastComputedInputSum;

    /**
     * Konstruktor
     * Initialisiert die Gewichte zufällig in einem Bereich
     * @param parent Der Layer, zu welchem das Neuron gehört
     * @param max_input_size Die maximale Größe der Eingaben, welche verarbeitet werden können (Anzahl von deren Gewichten)
     * @param neuron_id Identifikationsnummer, welche innerhalb des Layers eindeutig ist und dem Index des Neurons entspricht
     * @param min_weight_value Der kleinste Wert des Bereiches, in dem die Gewichte initialisiert werden
     * @param max_weight_value Der größte Wert des Bereiches, in dem die Gewichte initialisiert werden
     */
    public Neuron(Layer parent, int max_input_size, int neuron_id, double min_weight_value, double max_weight_value)
    {
        this.neuronId = neuron_id;
        this.layer = parent;
        this.weights = new double[max_input_size];

        double range = max_weight_value - min_weight_value;
        for (int i = 0; i < weights.length; i++)
        {
            this.weights[i] = ((Math.random() * range)) + min_weight_value;
        }
    }

    /**
     * Berechnet die Ausgabe des Neurons für eine übergebene Eingabe
     * @param x Eingabe
     * @return Ausgabe
     * @throws TooManyInputsException falls zu viele Eingaben vorhanden sind
     */
    public double compute(double[] x) throws TooManyInputsException
    {
        //check if there are enough weights for the input vector
        if (x.length > weights.length)
        {
            throw new TooManyInputsException(this.layer.getLayerId(), this.neuronId, x.length, this.weights.length);
        }

        //sum up all input values with their weights
        double sum = 0;
        for (int i = 0; i < x.length; i++)
        {
            sum += (x[i] * this.weights[i]);
        }
        sum += this.layer.getBias();
        this.lastComputedInputSum = sum;

        //System.out.println("Layer: " + this.layer.getLayerId() + " output: " + );

        //return the y-value for the sum
        return this.layer.getFunction().compute(sum);
    }

    public double getWeight(int index_of_neuron)
    {
        return this.weights[index_of_neuron];
    }

    /**
     * Für Backpropagation. Die Initialisierung erfolgt in den Subklassen, da sich die Algorithmen hier unterscheiden
     */
    public abstract void changeWeights();

    /**
     * Gibt ein Objekt mit Daten über das Objekt zurück
     * Nur für Debugging benötigt
     * @return Neuron-Data-Objekt mit entsprechenden Daten
     */
    public NeuronData getData()
    {
        NeuronData result = new NeuronData();
        result.id = this.neuronId;
        result.layer_id = this.layer.getLayerId();
        result.bias = this.layer.getBias();
        result.tolerance = this.layer.getTolerance();
        result.algorithm = this.layer.getFunction().getAlgorithmName();
        result.weights = new double[this.weights.length];
        System.arraycopy(this.weights, 0, result.weights, 0, weights.length);
        return result;
    }

    /**
     * Setzt die Gewichte manuell; für das Laden aus der Persistenten Speicherung erforderlich
     * @param weights die Gewichte
     */
    public void setWeights(double[] weights)
    {
        this.weights = weights;
    }

    //represents a status of the neuron with all its properties
    public static class NeuronData
    {
        public int id;
        public int layer_id;
        public double[] weights;
        public double bias;
        public double tolerance;
        public String algorithm;
    }
}
