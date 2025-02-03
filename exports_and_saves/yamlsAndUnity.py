using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using YamlDotNet.Serialization;
using System.IO;

public class HandMovement : MonoBehaviour
{
    public string yamlFilePath = "Assets/angles_data.yaml";
    public Transform thumb;
    public Transform index;
    public Transform middle;
    public Transform ring;
    public Transform pinky;

    private List<FrameData> frames;
    private int currentFrame = 0;

    void Start()
    {
        // Leer el archivo YAML
        var deserializer = new Deserializer();
        using (var reader = new StreamReader(yamlFilePath))
        {
            frames = deserializer.Deserialize<List<FrameData>>(reader);
        }

        // Iniciar la Corrutina para reproducir el movimiento
        StartCoroutine(PlayHandMovement());
    }

    IEnumerator PlayHandMovement()
    {
        while (true)
        {
            if (currentFrame < frames.Count)
            {
                ApplyAngles(frames[currentFrame].angles);
                currentFrame++;
                yield return new WaitForSeconds(0.033f); // Aproximadamente 30 FPS
            }
            else
            {
                currentFrame = 0;
            }
        }
    }

    void ApplyAngles(List<AngleData> angles)
    {
        foreach (var angle in angles)
        {
            switch (angle.finger)
            {
                case "thumb":
                    thumb.localRotation = Quaternion.Euler(0, 0, angle.angle);
                    break;
                case "index":
                    index.localRotation = Quaternion.Euler(0, 0, angle.angle);
                    break;
                case "middle":
                    middle.localRotation = Quaternion.Euler(0, 0, angle.angle);
                    break;
                case "ring":
                    ring.localRotation = Quaternion.Euler(0, 0, angle.angle);
                    break;
                case "pinky":
                    pinky.localRotation = Quaternion.Euler(0, 0, angle.angle);
                    break;
            }
        }
    }
}

public class FrameData
{
    public int frame { get; set; }
    public List<AngleData> angles { get; set; }
}

public class AngleData
{
    public string finger { get; set; }
    public float angle { get; set; }
}
