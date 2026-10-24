# Esquema de Arquitectura para Sistema de Análisis Documental con Visión

## 1. Arquitectura General

### Estructura Jerárquica

```
DocumentAnalyzer System
│
├── Orchestrator Agent (Principal)
│   ├── Reasoning: Analiza la solicitud del usuario
│   ├── Planning: Planifica los pasos necesarios
│   └── Action: Coordina subagentes y tools
│
├── Vision Subagent
│   ├── Vision Skill: Procesa imágenes/documentos
│   └── Tools: read (imágenes), webfetch (URLs)
│
├── Text Analysis Subagent
│   ├── NLP Skill: Análisis de texto
│   └── Tools: read (textos), webfetch
│
├── Knowledge Retrieval Subagent
│   ├── Search Skill: Búsqueda en base de conocimientos
│   └── Tools: websearch, glob, grep
│
└── Report Generation Subagent
    ├── Writing Skill: Generación de reportes
    └── Tools: write
```

### Flujo de Reasoning-Acting

```
User Request
    ↓
[REASONING] Agent principal analiza la solicitud
    ↓
[PLANNING] Define plan de análisis (qué subagentes, qué tools)
    ↓
[ACTING] Subagentes ejecutan con sus tools
    ↓
[REASONING] Agent evalúa resultados intermedios
    ↓
[ACTING] Itera/refina si es necesario
    ↓
[OUTPUT] Genera respuesta final
```

---

## 2. Estrategias para Extraer Estructura de Documentos

### 2.1. Resumen (Summarization)

#### Estrategia por Tamaño:

| Tamaño | Estrategia | Rationale |
|--------|-----------|-----------|
| **Pequeños** (<1 página) | **Extractivo completo** | Todo el contenido es relevante, no hay ruido |
| **Medianos** (1-5 páginas) | **Extractivo + Generativo** | Extraer secciones clave + generar conclusión |
| **Grandes** (>5 páginas) | **Hierárquico** | 1. Extraer estructura (títulos, secciones) 2. Resumir por sección 3. Sintetizar |

#### Flujo recomendado:

```python
def smart_summarize(documents, max_length=500):
    # Clasificar por tamaño
    small = [d for d in documents if len(d) < 1000]
    medium = [d for d in documents if 1000 <= len(d) < 5000]
    large = [d for d in documents if len(d) >= 5000]
    
    results = []
    
    for doc in documents:
        if doc.size < 1000:  # Pequeño
            summary = extractive_summary(doc)
        elif doc.size < 5000:  # Mediano
            summary = extractive_section_summary(doc)
        else:  # Grande
            sections = extract_structure(doc)
            section_summaries = [summarize_section(s) for s in sections]
            summary = hierarchical_synthesis(section_summaries)
        
        results.append({
            "document": doc,
            "summary": summary,
            "method": "extractive" if doc.size < 1000 else "hierarchical"
        })
    
    # Opcional: fusionar resúmenes si hay documentos muy similares
    return filter_similar_summaries(results, threshold=0.85)
```

### 2.2. Comparación (Comparison)

#### Estrategia por Escenario:

| Escenario | Estrategia | Herramientas |
|-----------|-----------|--------------|
| **Documentos similares** (mismo tema, diferente enfoque) | **Comparativa lado a lado** | Extraer puntos clave de cada uno, comparar punto por punto |
| **Documentos diferentes** (temas distintos) | **Análisis de diferencias** | Identificar qué tiene cada uno, qué falta en cada uno |
| **Evolución temporal** (versión 1 vs versión 2) | **Diferencial** | Detectar cambios, agregar, eliminar, modificar |

#### Flujo recomendado:

```python
def compare_documents(docs, comparison_type="comparativa"):
    # Extraer estructura comparable
    if comparison_type == "comparativa":
        # Extraer secciones equivalentes
        sections1 = extract_sections(docs[0])
        sections2 = extract_sections(docs[1])
        
        results = []
        for s1, s2 in zip(sections1, sections2):
            diff = compare_sections(s1, s2)
            results.append({"section": s1.title, "diff": diff})
    
    elif comparison_type == "diferencial":
        # Extraer contenido clave
        content1 = extract_key_points(docs[0])
        content2 = extract_key_points(docs[1])
        
        added = set(content2) - set(content1)
        removed = set(content1) - set(content2)
        modified = set(content1) & set(content2)  # Diferentes contextos
    
    return results
```

### 2.3. Auditoría (Auditing)

#### Estrategia por Escenario:

| Escenario | Estrategia | Qué verificar |
|-----------|-----------|---------------|
| **Consistencia** (mismo documento, múltiples versiones) | **Detección de cambios** | ¿Qué cambió? ¿Es significativo? |
| **Calidad** (un solo documento) | **Validación de estructura** | ¿Tiene todas las secciones requeridas? ¿Formato correcto? |
| **Integridad** (múltiples documentos relacionados) | **Verificación de consistencia** | ¿Se contradicen? ¿Referencias cruzadas válidas? |

#### Flujo recomendado:

```python
def audit_documents(documents, audit_type="consistencia"):
    if audit_type == "consistencia":
        # Comparar todas las parejas
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents):
                if i < j:
                    diff = compare_documents(doc1, doc2)
                    audit_results.append({
                        "doc1": doc1.id,
                        "doc2": doc2.id,
                        "changes": diff,
                        "similarity": calculate_similarity(doc1, doc2)
                    })
    
    elif audit_type == "calidad":
        # Validar contra checklist
        for doc in documents:
            checklist = validate_structure(doc)
            audit_results.append({
                "document": doc.id,
                "issues": checklist.issues,
                "score": checklist.score
            })
    
    elif audit_type == "integridad":
        # Verificar consistencia cruzada
        for doc in documents:
            references = extract_references(doc)
            for ref in references:
                if ref not in documents:
                    audit_results.append({
                        "document": doc.id,
                        "issue": f"Referencia no encontrada: {ref}"
                    })
    
    return audit_results
```

---

## 3. Estrategias para Extraer Estructura en Java

### 3.1. Extracción con Librerías Java (Capa 1 - Rápida)

#### Para PDF:

| Librería | Ventaja | Limitaciones |
|----------|---------|--------------|
| **Apache PDFBox** | Estándar de la industria, muy rápido, mantiene layout básico | No detecta tablas complejas, requiere configuración |
| **PDFBox + PDFTextStripper** | Extrae texto preservando estructura básica | Pierde formato complejo |
| **Apache Tika** | Soporta múltiples formatos, detecta metadatos | Menos preciso en layout |
| **iText (Community)** | Funciones avanzadas de layout | Licencia restrictiva (GPL) |

```java
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.pdmodel.PDPage;

public class PdfStructureExtractor {
    
    public DocumentStructure extract(String pdfPath) throws IOException {
        PDFTextStripper stripper = new PDFTextStripper();
        stripper.setKeepFormatting(true);
        
        DocumentStructure structure = new DocumentStructure();
        structure.setFilePath(pdfPath);
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(pdfPath), StandardCharsets.UTF_8))) {
            
            String text = stripper.getText(reader);
            structure.setText(text);
            
            // Extraer páginas
            PDFDocument pdf = new PDFDocument(new FileInputStream(pdfPath));
            for (int i = 0; i < pdf.getNumberOfPages(); i++) {
                structure.addPage(i, extractPageStructure(pdf.getPage(i)));
            }
        }
        
        return inferHierarchy(structure);
    }
    
    private PageStructure extractPageStructure(PDPage page) {
        PageStructure ps = new PageStructure();
        ps.setPageNumber(page.getNumberOfPages());
        
        // Extraer texto por renglones
        List<String> lines = extractLines(page);
        ps.setLines(lines);
        
        return ps;
    }
}
```

#### Para Word (.docx):

| Librería | Ventaja | Limitaciones |
|----------|---------|--------------|
| **Apache POI (XWPF)** | Estándar, muy completo, preserva estilos | Lento con docs grandes |
| **Docx4j** | Más rápido que POI, mejor para estructura | Menos funciones |

```java
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFTable;

public class DocxStructureExtractor {
    
    public DocumentStructure extract(String docxPath) throws IOException {
        DocumentStructure structure = new DocumentStructure();
        structure.setFilePath(docxPath);
        
        try (XWPFDocument doc = new XWPFDocument(new FileInputStream(docxPath))) {
            
            // Extraer párrafos con estilos
            List<ParagraphInfo> paragraphs = new ArrayList<>();
            for (XWPFParagraph para : doc.getParagraphs()) {
                String style = para.getParagraphStyle().getStyleId();
                int level = getHeadingLevel(style);
                
                paragraphs.add(new ParagraphInfo(
                    para.getText(),
                    style,
                    level
                ));
            }
            
            // Extraer tablas
            List<TableInfo> tables = new ArrayList<>();
            for (XWPFTable tbl : doc.getTables()) {
                List<List<String>> rows = new ArrayList<>();
                for (XWPFTableRow row : tbl.getRows()) {
                    List<String> rowData = new ArrayList<>();
                    for (XWPFTableCell cell : row.getTableCells()) {
                        rowData.add(cell.getText(0));
                    }
                    rows.add(rowData);
                }
                tables.add(new TableInfo(rows));
            }
            
            structure.setParagraphs(paragraphs);
            structure.setTables(tables);
        }
        
        return inferHierarchy(structure);
    }
    
    private int getHeadingLevel(String styleId) {
        if (styleId.contains("Heading1")) return 1;
        if (styleId.contains("Heading2")) return 2;
        if (styleId.contains("Heading3")) return 3;
        return 0; // Párrafo normal
    }
}
```

#### Para PPTX:

| Librería | Ventaja | Limitaciones |
|----------|---------|--------------|
| **Apache POI (XSLFX)** | Estándar, completo | Lento, complejo |
| **Slides** | Más simple, rápido | Menos funciones |

```java
import org.apache.poi.xslf.usermodel.XSLFSlide;

public class PptxStructureExtractor {
    
    public DocumentStructure extract(String pptxPath) throws IOException {
        DocumentStructure structure = new DocumentStructure();
        structure.setFilePath(pptxPath);
        
        try (XSLFSlide slide = new XSLFSlide(new FileInputStream(pptxPath))) {
            
            // Extraer diapositivas
            List<SlideInfo> slides = new ArrayList<>();
            for (XSLFSlide s : pptx.getSlides()) {
                SlideInfo si = new SlideInfo();
                si.setSlideNumber(s.getSlideNumber());
                si.setLayout(s.getLayoutType().toString());
                
                // Extraer título
                if (s.hasTitle()) {
                    si.setTitle(s.getTitle().getText());
                }
                
                // Extraer bullets
                si.setBulletPoints(extractBullets(s));
                
                // Extraer tablas e imágenes
                si.setTables(extractTables(s));
                si.setImages(extractImages(s));
                
                slides.add(si);
            }
            
            structure.setSlides(slides);
        }
        
        return inferHierarchy(structure);
    }
}
```

### 3.2. Estrategias de Código para Extraer Estructura

#### 1. Análisis de Patrones de Títulos (Regex-based)

```java
public class TitlePatternExtractor {
    
    public List<TitleInfo> extractTitles(String text) {
        List<TitleInfo> titles = new ArrayList<>();
        
        // Patrón 1: Títulos en mayúsculas con dos palabras
        Pattern pattern1 = Pattern.compile("([A-Z][a-z]+\\s+[A-Z][a-z]+)");
        Matcher matcher1 = pattern1.matcher(text);
        while (matcher1.find()) {
            titles.add(new TitleInfo(matcher1.group(), 1));
        }
        
        // Patrón 2: Títulos con guion mayor
        Pattern pattern2 = Pattern.compile("(?m)^([A-Z][A-Z\\s]+):");
        Matcher matcher2 = pattern2.matcher(text);
        while (matcher2.find()) {
            titles.add(new TitleInfo(matcher2.group().replace(":", ""), 1));
        }
        
        // Patrón 3: Títulos con números
        Pattern pattern3 = Pattern.compile("(?m)^([0-9]+\\.)\\s+([A-Z][a-z]+)");
        Matcher matcher3 = pattern3.matcher(text);
        while (matcher3.find()) {
            titles.add(new TitleInfo(matcher3.group(), 2));
        }
        
        return titles;
    }
}
```

#### 2. Detección de Jerarquía por Estilos/Clases

```java
public class StyleHierarchyExtractor {
    
    public DocumentHierarchy extractHierarchy(String text, Map<String, StyleInfo> styles) {
        DocumentHierarchy hierarchy = new DocumentHierarchy();
        hierarchy.setTitle("Documento");
        
        List<DocumentSection> sections = new ArrayList<>();
        String currentSection = "";
        int currentLevel = 0;
        
        for (String line : text.split("\n")) {
            StyleInfo style = styles.get(line);
            
            if (style != null && style.isHeading()) {
                // Crear nueva sección si cambiamos de nivel
                if (style.getLevel() > currentLevel) {
                    if (!currentSection.isEmpty()) {
                        sections.add(new DocumentSection(currentSection, currentLevel));
                    }
                    currentSection = line;
                    currentLevel = style.getLevel();
                }
            } else {
                // Añadir contenido a sección actual
                currentSection += line + " ";
            }
        }
        
        if (!currentSection.isEmpty()) {
            sections.add(new DocumentSection(currentSection, currentLevel));
        }
        
        hierarchy.setSections(sections);
        return hierarchy;
    }
}
```

#### 3. Detección de Tablas por Layout

```java
public class TableDetector {
    
    public List<TableInfo> detectTables(String text, List<String> lines) {
        List<TableInfo> tables = new ArrayList<>();
        
        // Detectar líneas separadoras (---, ===, ---, etc.)
        List<Integer> separatorLines = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (isSeparatorLine(line)) {
                separatorLines.add(i);
            }
        }
        
        // Agrupar líneas entre separadores como tablas
        for (int i = 0; i < separatorLines.size() - 1; i++) {
            int start = separatorLines.get(i) + 1;
            int end = separatorLines.get(i + 1);
            
            if (end - start >= 2) { // Mínimo 2 líneas para ser tabla
                List<List<String>> rows = new ArrayList<>();
                for (int j = start; j < end; j++) {
                    rows.add(Arrays.asList(lines.get(j).split("\\s+,\\s*")));
                }
                
                tables.add(new TableInfo(rows, "tabla_detectada"));
            }
        }
        
        return tables;
    }
    
    private boolean isSeparatorLine(String line) {
        String trimmed = line.trim();
        return trimmed.matches("^[\\-=]+[\\n\\r]$") || 
               trimmed.matches("^[|\\+\\-]+[\\n\\r]$");
    }
}
```

#### 4. Detección de Listas/Numeración

```java
public class ListDetector {
    
    public List<ListInfo> detectLists(String text) {
        List<ListInfo> lists = new ArrayList<>();
        
        // Detectar listas numeradas
        Pattern numberedPattern = Pattern.compile("(?m)^([0-9]+[\\.)\\s]+(.+)$");
        Matcher numberedMatcher = numberedPattern.matcher(text);
        
        List<String> numberedItems = new ArrayList<>();
        while (numberedMatcher.find()) {
            numberedItems.add(numberedMatcher.group(2));
        }
        
        if (!numberedItems.isEmpty()) {
            lists.add(new ListInfo("numerada", numberedItems));
        }
        
        // Detectar listas con bullets (-, *, •, etc.)
        Pattern bulletedPattern = Pattern.compile("(?m)^([•\\*\\-]\\s+.+)$");
        Matcher bulletedMatcher = bulletedPattern.matcher(text);
        
        List<String> bulletedItems = new ArrayList<>();
        while (bulletedMatcher.find()) {
            bulletedItems.add(bulletedMatcher.group(1));
        }
        
        if (!bulletedItems.isEmpty()) {
            lists.add(new ListInfo("bullets", bulletedItems));
        }
        
        return lists;
    }
}
```

#### 5. Extracción de Metadatos del Documento

```java
public class DocumentMetadataExtractor {
    
    public DocumentMetadata extractMetadata(String filePath) {
        DocumentMetadata metadata = new DocumentMetadata();
        
        // Extraer nombre del archivo
        metadata.setFileName(new File(filePath).getName());
        
        // Extraer extensión
        metadata.setExtension(filePath.split("\\.")[filePath.split("\\.").length - 1]);
        
        // Extraer tamaño
        metadata.setSize(new File(filePath).length());
        
        // Extraer fecha de modificación
        metadata.setLastModified(new File(filePath).lastModified());
        
        // Extraer autor (si está disponible en el parser)
        // metadata.setAuthor(parser.getAuthor());
        
        // Extraer fecha de creación (si está disponible)
        // metadata.setCreatedDate(parser.getCreatedDate());
        
        return metadata;
    }
}
```

#### 6. Detección de Encabezados y Pies de Página

```java
public class HeaderFooterDetector {
    
    public List<PageHeaderFooter> extractHeadersFooters(String text, int totalPages) {
        List<PageHeaderFooter> headersFooters = new ArrayList<>();
        
        // Extraer primeras 3 líneas como posibles encabezados
        String[] lines = text.split("\\n");
        
        for (int i = 0; i < Math.min(3, lines.length); i++) {
            String line = lines[i].trim();
            
            // Encabezado: línea corta, repetida en todas las páginas
            if (line.length() < 50 && isRepeatingLine(line, lines)) {
                headersFooters.add(new PageHeaderFooter(line, true, false));
            }
            
            // Pie de página: línea al final, con números de página
            if (i == lines.length - 1) {
                if (line.matches(".*\\d{1,3}.*")) {
                    headersFooters.add(new PageHeaderFooter(line, false, true));
                }
            }
        }
        
        return headersFooters;
    }
    
    private boolean isRepeatingLine(String line, String[] allLines) {
        int count = 0;
        for (String l : allLines) {
            if (l.trim().equals(line)) {
                count++;
            }
        }
        return count > 2; // Aparece en más de 2 páginas
    }
}
```

#### 7. Extracción de Referencias/Citas

```java
public class ReferenceExtractor {
    
    public List<ReferenceInfo> extractReferences(String text) {
        List<ReferenceInfo> references = new ArrayList<>();
        
        // Patrón para citas [1], [2], etc.
        Pattern citationPattern = Pattern.compile("\\[([0-9]+)\\]");
        Matcher citationMatcher = citationPattern.matcher(text);
        
        while (citationMatcher.find()) {
            references.add(new ReferenceInfo("[" + citationMatcher.group(1) + "]"));
        }
        
        // Patrón para referencias bibliográficas
        Pattern bibPattern = Pattern.compile("(?s)(?:^|\\n)(?:\\d+\\.)?([A-Z][a-z]+(?:,\\s+[A-Z][a-z]+)*)\\s+([0-9]{4})");
        Matcher bibMatcher = bibPattern.matcher(text);
        
        while (bibMatcher.find()) {
            references.add(new ReferenceInfo("bibliografía", bibMatcher.group(1), bibMatcher.group(2)));
        }
        
        return references;
    }
}
```

#### 8. Clasificador de Secciones por Contenido

```java
public class SectionClassifier {
    
    public Map<String, Integer> classifySections(String text) {
        Map<String, Integer> classifications = new HashMap<>();
        
        // Palabras clave para cada tipo de sección
        Map<String, String[]> keywords = new HashMap<>();
        keywords.put("introduccion", new String[]{"introducción", "introduct", "presentación", "antecedentes"});
        keywords.put("metodologia", new String[]{"metodología", "método", "procedimiento", "enfoque"});
        keywords.put("resultados", new String[]{"resultados", "hallazgos", "conclusiones", "encontramos"});
        keywords.put("conclusiones", new String[]{"conclusiones", "conclusión", "conclusion", "reflexión"});
        
        String[] lines = text.split("\\n");
        
        for (String line : lines) {
            for (Map.Entry<String, String[]> entry : keywords.entrySet()) {
                String sectionType = entry.getKey();
                String[] keywords = entry.getValue();
                
                for (String keyword : keywords) {
                    if (line.toLowerCase().contains(keyword)) {
                        classifications.put(sectionType, classifications.getOrDefault(sectionType, 0) + 1);
                        break;
                    }
                }
            }
        }
        
        return classifications;
    }
}
```

#### 9. Orquestador de Extracción de Estructura

```java
public class DocumentStructureExtractor {
    
    private final TextParser textParser;
    private final TitlePatternExtractor titleExtractor;
    private final StyleHierarchyExtractor styleExtractor;
    private final TableDetector tableDetector;
    private final ListDetector listDetector;
    private final DocumentMetadataExtractor metadataExtractor;
    private final HeaderFooterDetector headerFooterDetector;
    private final SectionClassifier sectionClassifier;
    
    public DocumentStructure extract(String filePath) {
        // Extraer texto con parser existente
        String text = textParser.extract(filePath);
        
        // Extraer metadatos
        DocumentMetadata metadata = metadataExtractor.extractMetadata(filePath);
        
        // Detectar estructura
        DocumentStructure structure = new DocumentStructure();
        structure.setMetadata(metadata);
        structure.setText(text);
        
        // Extraer títulos
        List<TitleInfo> titles = titleExtractor.extractTitles(text);
        structure.setTitles(titles);
        
        // Detectar tablas
        List<TableInfo> tables = tableDetector.detectTables(text);
        structure.setTables(tables);
        
        // Detectar listas
        List<ListInfo> lists = listDetector.detectLists(text);
        structure.setLists(lists);
        
        // Detectar encabezados/pies
        List<PageHeaderFooter> headersFooters = headerFooterDetector.extractHeadersFooters(text);
        structure.setHeadersFooters(headersFooters);
        
        // Inferir jerarquía
        DocumentHierarchy hierarchy = inferHierarchy(structure);
        structure.setHierarchy(hierarchy);
        
        return structure;
    }
    
    private DocumentHierarchy inferHierarchy(DocumentStructure structure) {
        DocumentHierarchy hierarchy = new DocumentHierarchy();
        hierarchy.setTitle(structure.getMetadata().getFileName().replace(".pdf", "").replace(".docx", ""));
        
        // Usar títulos principales como secciones de nivel 1
        for (TitleInfo title : structure.getTitles()) {
            if (title.getLevel() == 1) {
                hierarchy.addSection(new DocumentSection(title.getText(), 1));
            }
        }
        
        // Añadir contenido entre títulos
        // ...
        
        return hierarchy;
    }
}
```

---

## 4. Estrategias de Búsqueda de Documentos

### 4.1. Búsqueda Híbrida (BM25 + Vector)

```java
public class HybridSearch {
    
    public List<Document> search(String query, double bm25Weight, double vectorWeight, int topK) {
        // Buscar con BM25
        List<Document> bm25Results = bm25Search(query, topK * 2);
        
        // Buscar con vectores
        List<Document> vectorResults = vectorSearch(query, topK * 2);
        
        // Reciprocal Rank Fusion (RRF)
        Map<String, Double> scores = new HashMap<>();
        double k = 60;
        
        for (int i = 0; i < bm25Results.size(); i++) {
            String docId = bm25Results.get(i).getId();
            scores.put(docId, scores.getOrDefault(docId, 0.0) + bm25Weight / (i + k));
        }
        
        for (int i = 0; i < vectorResults.size(); i++) {
            String docId = vectorResults.get(i).getId();
            scores.put(docId, scores.getOrDefault(docId, 0.0) + vectorWeight / (i + k));
        }
        
        // Ordenar y devolver topK
        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(entry -> bm25Results.stream()
                .filter(d -> d.getId().equals(entry.getKey()))
                .findFirst()
                .orElse(vectorResults.stream()
                    .filter(d -> d.getId().equals(entry.getKey()))
                    .findFirst()
                    .orElse(null)))
            .collect(Collectors.toList());
    }
}
```

### 4.2. Búsqueda Exacta (Solo BM25)

```java
public class BM25Search {
    
    public List<Document> search(String query, int topK) {
        // Calcular scores BM25 para cada documento
        Map<String, Double> scores = new HashMap<>();
        
        for (Document doc : documents) {
            double score = calculateBM25Score(doc, query);
            scores.put(doc.getId(), score);
        }
        
        // Ordenar y devolver topK
        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(entry -> documents.stream()
                .filter(d -> d.getId().equals(entry.getKey()))
                .findFirst()
                .orElse(null))
            .collect(Collectors.toList());
    }
    
    private double calculateBM25Score(Document doc, String query) {
        double idf = calculateIDF(query, doc);
        double tf = calculateTF(query, doc);
        return idf * tf;
    }
}
```

### 4.3. Búsqueda Semántica (Solo Vectores)

```java
public class VectorSearch {
    
    public List<Document> search(String query, int topK) {
        // Obtener embedding de la query
        Embedding queryEmbedding = embeddingModel.embed(query);
        
        // Buscar vecinos más cercanos
        List<Document> results = vectorStore.search(queryEmbedding, topK);
        
        return results;
    }
}
```

---

## 5. Schema de Tools

```json
{
  "extract_text": {
    "input": {"filePath": "string"},
    "output": {"text": "string", "metadata": "object"}
  },
  "extract_structure": {
    "input": {"text": "string"},
    "output": {"sections": "list", "hierarchy": "tree"}
  },
  "summarize": {
    "input": {"text": "string", "method": "string", "max_length": "int"},
    "output": {"summary": "string", "method": "string"}
  },
  "compare": {
    "input": {"doc1": "string", "doc2": "string", "type": "string"},
    "output": {"differences": "list", "similarity": "float"}
  },
  "audit": {
    "input": {"documents": "list", "type": "string"},
    "output": {"issues": "list", "score": "float"}
  },
  "bm25_search": {
    "input": {"query": "string", "top_k": "int"},
    "output": {"documents": "list", "scores": "dict"}
  },
  "vector_search": {
    "input": {"query": "string", "top_k": "int"},
    "output": {"documents": "list", "scores": "dict"}
  },
  "hybrid_search": {
    "input": {"query": "string", "bm25_weight": "float", "vector_weight": "float", "top_k": "int"},
    "output": {"documents": "list", "scores": "dict", "method": "string"}
  }
}
```

---

## 6. Recomendaciones Finales

### 6.1. Para Resumen

| Tamaño | Estrategia |
|--------|-----------|
| Pequeño (<1KB) | Extractivo completo |
| Mediano (1-5KB) | Extractivo por secciones |
| Grande (>5KB) | Hierárquico (secciones → síntesis) |

### 6.2. Para Comparación

| Escenario | Estrategia |
|-----------|-----------|
| Documentos similares | Comparativa lado a lado |
| Documentos diferentes | Análisis de diferencias |
| Evolución temporal | Diferencial |

### 6.3. Para Auditoría

| Escenario | Estrategia |
|-----------|-----------|
| Consistencia | Detección de cambios |
| Calidad | Validación de estructura |
| Integridad | Verificación de consistencia |

### 6.4. Para Extracción de Estructura

| Formato | Estrategia Principal | Estrategias Complementarias |
|---------|---------------------|---------------------------|
| PDF | Apache PDFBox + Regex | Tablas, Headers |
| Word | Apache POI + Estilos | Tablas, Listas |
| PPTX | Apache POI + Layout | Títulos, Bullets |

### 6.5. Para Búsqueda

| Caso de uso | Estrategia |
|-------------|-----------|
| Búsqueda exacta | Solo BM25 |
| Búsqueda semántica | Solo Vectores |
| La mayoría de casos | Híbrida (RRF) |

---

## 7. Resumen de Enfoques

### Enfoque por Capas:

```
Capa 1: Extracción de Texto (Parser)
    ↓
Capa 2: Extracción de Estructura (Código + LLM opcional)
    ↓
Capa 3: Análisis (Resumen/Comparación/Auditoría)
    ↓
Capa 4: Búsqueda (BM25, Vector, Híbrido)
    ↓
Capa 5: Generación de Reporte
```

### Enfoque Híbrido Recomendado:

1. **Extracción con código** para documentos estructurados (PDF, Word, PPTX)
2. **LLM** para inferencia de jerarquía en documentos complejos
3. **Validación con LLM** solo si la confianza del código es baja (<0.7)
4. **Cache** los resultados para evitar llamadas redundantes al LLM

---

## 8. Implementación en Java

### Clase Principal:

```java
public class DocumentAnalyzerSystem {
    
    private final LLMClient llmClient;
    private final DocumentStructureExtractor structureExtractor;
    private final DocumentAnalyzer analyzer;
    
    public DocumentAnalysis analyze(String docPath, AnalysisType type) {
        // Extraer estructura
        DocumentStructure structure = structureExtractor.extract(docPath);
        
        // Extraer contenido
        String content = extractContent(docPath);
        
        // Aplicar operación según tipo
        switch (type) {
            case SUMMARIZE:
                return analyzer.summarize(structure, content);
            case COMPARE:
                return analyzer.compare(structure, content);
            case AUDIT:
                return analyzer.audit(structure, content);
        }
        
        return null;
    }
}
```

---

## 9. Comparativa de Enfoques

| Enfoque | Velocidad | Precisión | Costo | Recomendado para |
|---------|----------|-----------|-------|-----------------|
| **Solo código** | ⚡⚡⚡ Muy rápido | ⭐⭐⭐ Para docs estructurados | $ | PDFs, Word simples, PPTs |
| **Solo LLM** | 🐌 Lento | ⭐⭐⭐⭐⭐ Para todo | $$$ | Docs complejos, no estructurados |
| **Híbrido** | ⚡ Rápido | ⭐⭐⭐⭐ Para todo | $$ | **Recomendado** |

---

## 10. Conclusión

### Recomendación Final:

**Usa enfoque híbrido con:**
1. **Código** para extraer la estructura básica (títulos, tablas, secciones)
2. **LLM** solo cuando el código falla o para refinar jerarquías complejas
3. **Cache** los resultados para no llamar al LLM cada vez

### Para tu caso específico:

- **Extracción de estructura**: Híbrida (código primero, LLM si falla)
- **Resumen**: Hierárquico para todos los tamaños
- **Comparación**: 3 niveles (estructural, contenido, calidad)
- **Auditoría**: Checklist paramétrico
- **Búsqueda**: Híbrida con RRF como default

---

*Generado para sistema de análisis documental con visión*