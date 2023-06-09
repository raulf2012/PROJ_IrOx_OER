# Playing with the features used, whether they create better models or not
---



### **No electronic descriptors (9 features)**

```python
'active_o_metal_dist', 'angle_O_Ir_surf_norm', 'ir_o_mean', 'ir_o_std', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

7 PCA components are ideal with an MAE of 0.1841



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed angle_O_Ir_surf_norm (8 features)**

```python
'active_o_metal_dist', 'ir_o_mean', 'ir_o_std', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

8 PCA components are ideal with an MAE of 0.1818

<span style="color:blue">2 meV improvment vs prev. model</span>



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed ir_o_std (7 features)**

```python
'active_o_metal_dist', 'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```
7 PCA components are ideal with an MAE of 0.1832

<span style="color:blue">Model become worse again, interestingly as good as the full model with all features</span>



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed active_o_metal_dist (6 features)**

```python
'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

6 PCA components are ideal with an MAE of 0.1813

<span style="color:blue">Model become worse again, interestingly as good as the full model with all features</span>


<p>&nbsp;</p>
<p>&nbsp;</p>

***
***

<p>&nbsp;</p>

### **Removed bulk_oxid_state (5 features)**

```python
'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

5 PCA components are ideal with an MAE of 0.1852

<span style="color:blue">TEMP TEMP</span>



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed volume_pa (5 features)**

```python
'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

5 PCA components are ideal with an MAE of 0.188

<span style="color:blue">TEMP TEMP</span>



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed dH_bulk (5 features)**

```python
'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

5 PCA components are ideal with an MAE of 0.185

<span style="color:blue">TEMP TEMP</span>



<p>&nbsp;</p>
<p>&nbsp;</p>

### **Removed effective_ox_state (5 features)**

```python
'ir_o_mean', 'octa_vol', 'dH_bulk', 'volume_pa', 'bulk_oxid_state', 'effective_ox_state'
```

5 PCA components are ideal with an MAE of 0.2269

<span style="color:blue">Highest drop in MAE, this feature is essential</span>

<p>&nbsp;</p>
<p>&nbsp;</p>

***
***

# Testing which single specific structural feature is the best

* 'active_o_metal_dist'
* 'angle_O_Ir_surf_norm'
* 'ir_o_mean'
* 'ir_o_std'
* 'octa_vol'


### **active_o_metal_dist (5 features)**

```python
active_o_metal_dist, dH_bulk, volume_pa, bulk_oxid_state, effective_ox_state
```

5 PCA components are ideal with an MAE of **0.1944**

<span style="color:blue">TEMP TEMP</span>



### **angle_O_Ir_surf_norm (5 features)**

```python
angle_O_Ir_surf_norm, dH_bulk, volume_pa, bulk_oxid_state, effective_ox_state
```

5 PCA components are ideal with an MAE of **0.2019**

<span style="color:blue">TEMP TEMP</span>



### **ir_o_mean (5 features)**

```python
ir_o_mean, dH_bulk, volume_pa, bulk_oxid_state, effective_ox_state
```

5 PCA components are ideal with an MAE of **0.1859**

<span style="color:blue">TEMP TEMP</span>



### **ir_o_std (5 features)**

```python
ir_o_std, dH_bulk, volume_pa, bulk_oxid_state, effective_ox_state
```

5 PCA components are ideal with an MAE of **0.1787**

<span style="color:blue">Wow, best one so far</span>



### **octa_vol (5 features)**

```python
ir_o_std, dH_bulk, volume_pa, bulk_oxid_state, effective_ox_state
```

5 PCA components are ideal with an MAE of **0.1851**

<span style="color:blue">Wow, best one so far</span>



