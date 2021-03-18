/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test.demo;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.net.NeuralNetSaver;
import de.jdreisvogt.sandbox.neuralnet.test.GUI;
import de.jdreisvogt.sandbox.neuralnet.test.MnistLoader;
import org.json.simple.parser.ParseException;

/**
 * Demo-Programm #1
 * Läd ein KNN und gibt eine bestimmte Anzahl von Testdatensätzen und deren Ergebnisse graphisch aus
 * Zur Ausführung muss die unter /code/Datenbank-Erstellen/datenbank.sql gespeicherte Datenbank erstellt und unter
 * folgenden Parametern verfügbar sein:
 * Name: 'Facharbeit', Port: 3306, SQL-Dialekt: MYSQL, Benutzername: 'root', Passwort: ''
 */
public class Demo1
{

    /**
     * Version mit Voreinstellungen, nur mit existentem Netz unter diesem Pfad erreichbar
     * @param args ignored
     */
    public static void main (String[] args)
    {
        execute("C:\\Users\\jonat\\Desktop\\Facharbeit\\Facharbeit - Logs\\Sigmoid [500; 200; 10; 10]\\net (1000R lr. 0.0001).nnet", 20);
    }

    /**
     * Version mit Parametern
     * @param path Der Pfad zu der .nnet Datei, welche das KNN enthält. Diese sollte von der NeuralNetSave-Klasse erstellt worden sein
     * @param size Die Anzahl von Datensätzen, die ausgegeben werden soll.
     */
    public static void execute (String path, int size)
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
            return;
        }
        MnistLoader.DataSet[] test = source.getPart(41999 - size, 41999);
        System.out.println("Daten wurden geladen");

        NeuralNet net;
        try
        {
            net = NeuralNetSaver.LoadNet(path);
        } catch (ParseException e)
        {
            System.out.println("Netz kann nicht geladen werden: Die Datei existiert nicht oder ist ungültig");
            return;
        }
        System.out.println("Netz wurde geladen");

        for (int i = 0; i < 20; i++)
        {
            GUI gui = new GUI(test[i].correctValue);
            double[][] pixel = new double[28][28];
            for (int j = 0; j < 28; j++)
            {
                double[] row = new double[28];
                System.arraycopy(test[i].x, (j * 28), row, 0, 28);
                pixel[j] = row;
            }
            gui.setPixel(pixel);
            gui.setResults(net.compute(test[i].x));
        }

        System.out.println("Daten wurden ausgegeben");
    }
}
